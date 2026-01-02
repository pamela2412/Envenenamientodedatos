import os
import json
import h5py
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader

# =========================
# CONFIG GENERAL
# =========================
OUT_DIR = "GTSRBzip"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5 = os.path.join(OUT_DIR, "gtsrb-64x64-sub18-clean_train_test.h5")

# Carpeta con modelos SIG a analizar
MODELS_DIR = "Modelosenvenenadossig"   # 👈 AQUÍ tu carpeta con .pth

BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =========================
# CONFIG NC-SIG (detección)
# =========================
TAU_SUCC = 0.40       # umbral de éxito por clase (40%)
GLOBAL_ASR_MIN = 0.3  # umbral global para declarar modelo backdooreado (30%)

DELTA_GRID = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
FREQ_GRID = [2, 3, 4, 5, 6]

# =========================
# HYPERPARAMS DEFENSA SIG
# =========================
DEFENSE_EPOCHS = 15
DEFENSE_LR = 1e-3
DEFENSE_ALPHA = 1.0

DEFENSE_OUT_DIR = os.path.join(OUT_DIR, "neural_cleanse_sig_defended_models")
os.makedirs(DEFENSE_OUT_DIR, exist_ok=True)

# =========================
# UTILS
# =========================

def to_tensor_x(arr):
    """
    arr: numpy (N, H, W, 3) en [0,1]
    -> tensor (N, 3, H, W) en [-1,1]
    """
    t = torch.tensor(arr.transpose(0, 3, 1, 2), dtype=torch.float32)
    t = (t - 0.5) * 2.0
    return t


def plant_sin_trigger(img, delta=20, f=6):
    """
    Implementación estilo SIG:
      x_b = clip(x + v)
    donde v(i,j) = delta * sin(2π j f / m)
    img: uint8 (H,W,3) en [0,255]
    """
    img = np.float32(img)
    pattern = np.zeros_like(img, dtype=np.float32)

    H, W, C = pattern.shape
    m = W  # ancho

    for i in range(H):
        for j in range(W):
            value = delta * np.sin(2 * np.pi * j * f / m)
            for k in range(C):
                pattern[i, j, k] = value

    # Suma aditiva, sin mezcla con alpha
    img_poison = img + pattern

    # Clipping a rango válido
    img_poison = np.clip(img_poison, 0, 255).astype(np.uint8)
    return img_poison


def build_resnet18(num_classes):
    """
    Debe ser compatible con el modelo que usaste para SIG.
    """
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate_sig_success_for_class(
    model,
    x_test,
    y_test,
    target_class,
    delta,
    freq,
    device="cuda",
    batch_size=128,
):
    """
    Aplica trigger SIG a TODAS las imágenes de test que NO son de la clase target_class,
    y mide la tasa de éxito: % de predicciones == target_class.
    """
    model.eval()

    # Filtrar test donde etiqueta != target_class
    mask = (y_test != target_class)
    x_other = x_test[mask].copy()
    y_other = y_test[mask]
    n_other = len(y_other)

    if n_other == 0:
        return 0.0

    # Aplicar trigger a TODO x_other (con este delta,freq)
    for i in range(n_other):
        img = (x_other[i] * 255.0).astype(np.uint8)
        img_p = plant_sin_trigger(img, delta=delta, f=freq)
        x_other[i] = img_p.astype(np.float32) / 255.0

    # Pasar a tensores
    x_other_tensor = to_tensor_x(x_other)
    y_other_tensor = torch.tensor(y_other, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(x_other_tensor, y_other_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    asr_num, asr_den = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            # Cuántas se van a la clase objetivo "target_class"
            asr_num += (preds == target_class).sum().item()
            asr_den += labels.size(0)

    asr = asr_num / max(1, asr_den)
    return asr


def fine_tune_with_sig_defense(
    model_path,
    delta,
    freq,
    x_train,
    y_train,
    num_classes,
    device,
    out_path,
    epochs=DEFENSE_EPOCHS,
    lr=DEFENSE_LR,
    alpha=DEFENSE_ALPHA,
    batch_size=BATCH_SIZE,
):
    """
    Fine-tuning de defensa para ataques SIG:
      - model_path: modelo envenenado (.pth)
      - delta, freq: parámetros del trigger SIG estimado para la clase sospechosa
      - x_train, y_train: datos LIMPIOS en [0,1] (N,H,W,3), (N,)
      - out_path: ruta donde guardar el modelo defendido
    """

    print(f"\n[DEF-SIG] Fine-tuning defensa SIG para modelo: {os.path.basename(model_path)}")
    print(f"[DEF-SIG] Usando delta={delta}, freq={freq}")
    print(f"[DEF-SIG] Guardando modelo defendido en: {out_path}")

    # 1) Cargar modelo envenenado
    model = build_resnet18(num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.train()

    # 2) Dataset de entrenamiento: mantenemos x en [0,1] y aplicamos trigger en numpy
    x_train = x_train.astype(np.float32)  # [0,1]
    y_train = y_train.astype(np.int64)

    # TensorDataset con x en [0,1] pero en formato (C,H,W)
    x_train_chw = np.transpose(x_train, (0, 3, 1, 2))  # (N,3,H,W)
    ds_train = TensorDataset(
        torch.from_numpy(x_train_chw),          # todavía en [0,1]
        torch.from_numpy(y_train),
    )
    loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for ep in range(epochs):
        total_loss = 0.0
        total_steps = 0
        correct_clean = 0
        correct_sig = 0
        total_clean = 0
        total_sig = 0

        for x01_chw, y in loader:
            # x01_chw: (B,3,H,W) en [0,1]
            x01_chw = x01_chw.to(device)
            y = y.to(device)

            bs = x01_chw.size(0)

            # ----- limpio -----
            x_clean = (x01_chw - 0.5) * 2.0          # normalización [-1,1]
            logits_clean = model(x_clean)
            loss_clean = criterion(logits_clean, y)

            # ----- con trigger SIG -----
            # Pasamos a numpy para aplicar plant_sin_trigger
            x01_np = x01_chw.detach().cpu().numpy()   # (B,3,H,W)
            x01_np = np.transpose(x01_np, (0, 2, 3, 1))  # (B,H,W,3)

            x_sig_np = np.empty_like(x01_np, dtype=np.float32)
            for i in range(bs):
                img_uint8 = (x01_np[i] * 255.0).astype(np.uint8)
                img_p = plant_sin_trigger(img_uint8, delta=delta, f=freq)
                x_sig_np[i] = img_p.astype(np.float32) / 255.0  # back to [0,1]

            x_sig_chw = np.transpose(x_sig_np, (0, 3, 1, 2))  # (B,3,H,W)
            x_sig_chw = torch.from_numpy(x_sig_chw).to(device)

            x_sig = (x_sig_chw - 0.5) * 2.0   # normalización [-1,1]
            logits_sig = model(x_sig)
            loss_sig = criterion(logits_sig, y)

            loss = loss_clean + alpha * loss_sig

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            with torch.no_grad():
                pred_clean = logits_clean.argmax(1)
                pred_sig = logits_sig.argmax(1)
                correct_clean += (pred_clean == y).sum().item()
                correct_sig += (pred_sig == y).sum().item()
                total_clean += bs
                total_sig += bs

        avg_loss = total_loss / max(total_steps, 1)
        acc_clean = 100.0 * correct_clean / max(total_clean, 1)
        acc_sig = 100.0 * correct_sig / max(total_sig, 1)

        print(
            f"[DEF-SIG] Epoch {ep+1}/{epochs} | "
            f"loss={avg_loss:.4f} | acc_clean={acc_clean:.2f}% | acc_sig={acc_sig:.2f}%"
        )

    # 3) Guardar modelo defendido
    torch.save(model.state_dict(), out_path)
    print(f"[DEF-SIG] Modelo defendido guardado en: {out_path}")
    return out_path


def evaluate_model_sig(
    model,
    x_test,
    y_test,
    target_class,
    delta,
    freq,
    device=device,
    batch_size=BATCH_SIZE,
):
    """
    Evalúa:
      - ACC en test limpio
      - ASR en test con trigger SIG hacia target_class
    """
    model.eval()

    # ---- ACC limpia ----
    x_clean_tensor = to_tensor_x(x_test)
    y_tensor = torch.tensor(y_test, dtype=torch.long)

    loader_clean = DataLoader(
        TensorDataset(x_clean_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    correct_clean, total_clean = 0, 0
    with torch.no_grad():
        for imgs, labels in loader_clean:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct_clean += (preds == labels).sum().item()
            total_clean += labels.size(0)

    acc_clean = correct_clean / max(1, total_clean)

    # ---- ASR con trigger SIG ----
    x_bd = x_test.copy()
    idx_nontarget = np.where(y_test != target_class)[0]

    for i in idx_nontarget:
        img = (x_bd[i] * 255.0).astype(np.uint8)
        img_p = plant_sin_trigger(img, delta=delta, f=freq)
        x_bd[i] = img_p.astype(np.float32) / 255.0

    x_bd_tensor = to_tensor_x(x_bd)

    loader_bd = DataLoader(
        TensorDataset(x_bd_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    total_nontarget, total_to_target = 0, 0
    with torch.no_grad():
        for imgs, labels in loader_bd:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            mask = (labels != target_class)
            if mask.sum() == 0:
                continue

            preds_nt = preds[mask]
            labels_nt = labels[mask]
            total_nontarget += labels_nt.size(0)
            total_to_target += (preds_nt == target_class).sum().item()

    asr = total_to_target / max(1, total_nontarget)

    return acc_clean, asr, correct_clean, total_clean, total_to_target, total_nontarget


# =========================
# MAIN
# =========================

def main():
    # --------------------------
    # 1) Cargar dataset limpio
    # --------------------------
    if not os.path.exists(OUT_H5):
        raise FileNotFoundError(
            f"No se encontró {OUT_H5}. Genera primero el H5 limpio."
        )

    print(f"\n📂 Cargando dataset limpio desde: {OUT_H5}")
    with h5py.File(OUT_H5, "r") as f:
        x_train = np.array(f["x_train"])   # [0,1]
        y_train = np.array(f["y_train"])
        x_test = np.array(f["x_test"])
        y_test = np.array(f["y_test"])
        keep_classes = np.array(f["keep_classes"])

    num_classes = len(np.unique(y_train))
    print(f"Clases remapeadas: {num_classes}")
    print("keep_classes (IDs originales GTSRB en orden remapeado):")
    print(keep_classes.tolist())

    # --------------------------
    # 2) Listar modelos a analizar
    # --------------------------
    model_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pth")
    ]
    model_files.sort()

    if not model_files:
        raise RuntimeError(f"No se encontraron modelos .pth en {MODELS_DIR}")

    print("\nModelos encontrados:")
    for mf in model_files:
        print("  -", mf)

    all_results = []

    # --------------------------
    # 3) Analizar cada modelo
    # --------------------------
    for mf in model_files:
        MODEL_PATH = os.path.join(MODELS_DIR, mf)
        MODEL_NAME = os.path.splitext(mf)[0]

        print("\n" + "="*70)
        print(f"🧠 Analizando modelo SIG: {MODEL_NAME}")
        print("="*70)

        model = build_resnet18(num_classes=num_classes).to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # --------------------------
        # 3.1 NC-SIG: búsqueda por clase
        # --------------------------
        per_class = []
        global_best_success = 0.0

        print("\n========= NC-SIG: búsqueda por clase =========")
        for c in range(num_classes):
            print(f"\n--- Clase {c} ---")
            best_success_c = 0.0
            best_delta_c = None
            best_freq_c = None
            min_delta_for_tau_c = None  # el delta más bajo que supera TAU_SUCC

            for delta in DELTA_GRID:
                for freq in FREQ_GRID:
                    asr_c = evaluate_sig_success_for_class(
                        model,
                        x_test,
                        y_test,
                        target_class=c,
                        delta=delta,
                        freq=freq,
                        device=device,
                        batch_size=BATCH_SIZE,
                    )
                    print(
                        f"  c={c} | delta={delta:4.1f}, f={freq} "
                        f"-> success={asr_c*100:.2f}%"
                    )

                    # Actualizar mejor ASR para esta clase
                    if asr_c > best_success_c:
                        best_success_c = asr_c
                        best_delta_c = delta
                        best_freq_c = freq

                    # Si pasa TAU_SUCC, miramos si es el delta mínimo que lo logra
                    if asr_c >= TAU_SUCC:
                        if min_delta_for_tau_c is None or delta < min_delta_for_tau_c:
                            min_delta_for_tau_c = delta

            # Actualizar global
            if best_success_c > global_best_success:
                global_best_success = best_success_c

            print(
                f"  >>> Clase {c} resumen: best_success={best_success_c*100:.2f}% "
                f"@ delta={best_delta_c}, f={best_freq_c}, "
                f"min_delta_for_tau={min_delta_for_tau_c}"
            )

            per_class.append(
                {
                    "class": int(c),
                    "best_success": float(best_success_c),
                    "best_delta": float(best_delta_c) if best_delta_c is not None else None,
                    "best_freq": int(best_freq_c) if best_freq_c is not None else None,
                    "min_delta_for_tau": float(min_delta_for_tau_c)
                    if min_delta_for_tau_c is not None
                    else None,
                }
            )

        # --------------------------
        # 3.2 Decidir si está backdooreado y clase sospechosa
        # --------------------------
        print("\n========= RESUMEN NC-SIG =========")
        print(f"GLOBAL best_success (max sobre todas las clases): {global_best_success*100:.2f}%")

        is_backdoored = global_best_success >= GLOBAL_ASR_MIN
        print(f"is_backdoored (según GLOBAL_ASR_MIN={GLOBAL_ASR_MIN*100:.1f}%): {is_backdoored}")

        # Score por clase:
        scores = []
        for pc in per_class:
            best_succ = pc["best_success"]
            min_delta_for_tau = pc["min_delta_for_tau"]

            if min_delta_for_tau is not None:
                score = float(min_delta_for_tau) / max(float(best_succ), 1e-3)
            else:
                score = 1e6
            scores.append(score)

        scores = np.array(scores, dtype=np.float64)
        median_score = float(np.median(scores))
        dev = np.abs(scores - median_score)
        mad_score = float(np.median(dev) + 1e-12)

        anomaly_indices = (1.4826 * dev / mad_score).tolist()

        suspect_classes = []
        if is_backdoored:
            c_star = int(np.argmin(scores))
            suspect_classes = [c_star]
            print(f"Clase sospechosa (score mínimo): {c_star}")
        else:
            print("Modelo considerado limpio según GLOBAL_ASR_MIN; no se marcan sospechosos.")
            c_star = None  # por seguridad

        # --------------------------
        # 3.3 Si está backdooreado, aplicar defensa y evaluar
        # --------------------------
        defended_model_path = None

        # Métricas por defecto (por si no se aplica defensa)
        acc_clean_orig = asr_orig = acc_clean_def = asr_def = None
        cc_o = tot_o = tt_o = tn_o = cc_d = tot_d = tt_d = tn_d = None

        if is_backdoored and suspect_classes:
            pc_star = per_class[c_star]
            best_delta_c = pc_star["best_delta"]
            best_freq_c = pc_star["best_freq"]

            if best_delta_c is None or best_freq_c is None:
                print(f"[DEF-SIG] {MODEL_NAME}: no se pudo obtener (delta,freq) válidos.")
            else:
                defended_model_path = os.path.join(
                    DEFENSE_OUT_DIR,
                    f"{MODEL_NAME}_sig_defended.pth"
                )

                # Fine-tuning de defensa
                fine_tune_with_sig_defense(
                    model_path=MODEL_PATH,
                    delta=best_delta_c,
                    freq=best_freq_c,
                    x_train=x_train,
                    y_train=y_train,
                    num_classes=num_classes,
                    device=device,
                    out_path=defended_model_path,
                    epochs=DEFENSE_EPOCHS,
                    lr=DEFENSE_LR,
                    alpha=DEFENSE_ALPHA,
                    batch_size=BATCH_SIZE,
                )

                # Evaluar antes y después con el mismo (delta,freq) y clase sospechosa
                print("\n========= EVALUACIÓN ANTES / DESPUÉS (SIG) =========")

                # Modelo original
                model_orig = build_resnet18(num_classes=num_classes).to(device)
                model_orig.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                acc_clean_orig, asr_orig, cc_o, tot_o, tt_o, tn_o = evaluate_model_sig(
                    model_orig,
                    x_test,
                    y_test,
                    target_class=c_star,
                    delta=best_delta_c,
                    freq=best_freq_c,
                    device=device,
                    batch_size=BATCH_SIZE,
                )

                # Modelo defendido
                model_def = build_resnet18(num_classes=num_classes).to(device)
                model_def.load_state_dict(torch.load(defended_model_path, map_location=device))
                acc_clean_def, asr_def, cc_d, tot_d, tt_d, tn_d = evaluate_model_sig(
                    model_def,
                    x_test,
                    y_test,
                    target_class=c_star,
                    delta=best_delta_c,
                    freq=best_freq_c,
                    device=device,
                    batch_size=BATCH_SIZE,
                )

                print(f"\nResultados usando clase sospechosa c*={c_star}, delta={best_delta_c}, freq={best_freq_c}:")
                print(f"  Modelo POISONED  - ACC clean: {acc_clean_orig*100:.2f}% ({cc_o}/{tot_o}) "
                      f"| ASR_SIG: {asr_orig*100:.2f}% ({tt_o}/{tn_o})")
                print(f"  Modelo DEFENDED  - ACC clean: {acc_clean_def*100:.2f}% ({cc_d}/{tot_d}) "
                      f"| ASR_SIG: {asr_def*100:.2f}% ({tt_d}/{tn_d})")

        else:
            print("\n[DEF-SIG] No se aplica defensa porque el modelo no fue marcado como backdooreado.")

        # --------------------------
        # 3.4 Guardar resumen en JSON para este modelo
        # --------------------------
        result = {
            "model_name": MODEL_NAME,
            "model_path": MODEL_PATH,
            "num_classes": int(num_classes),
            "TAU_SUCC": float(TAU_SUCC),
            "GLOBAL_ASR_MIN": float(GLOBAL_ASR_MIN),
            "global_best_success": float(global_best_success),
            "median_score": median_score,
            "mad_score": mad_score,
            "anomaly_indices": anomaly_indices,
            "suspect_classes": suspect_classes,
            "per_class": per_class,
            "defended_model_path": defended_model_path,
            "eval_sig": {
                "suspect_class": int(c_star) if c_star is not None else None,
                "acc_clean_poisoned": float(acc_clean_orig) if acc_clean_orig is not None else None,
                "asr_sig_poisoned": float(asr_orig) if asr_orig is not None else None,
                "acc_clean_defended": float(acc_clean_def) if acc_clean_def is not None else None,
                "asr_sig_defended": float(asr_def) if asr_def is not None else None,
                "clean_counts_poisoned": [int(cc_o), int(tot_o)] if cc_o is not None else None,
                "bd_counts_poisoned": [int(tt_o), int(tn_o)] if tt_o is not None else None,
                "clean_counts_defended": [int(cc_d), int(tot_d)] if cc_d is not None else None,
                "bd_counts_defended": [int(tt_d), int(tn_d)] if tt_d is not None else None,
            },
        }

        out_json = os.path.join(OUT_DIR, f"nc_sig_detection_and_defense_{MODEL_NAME}.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[NC-SIG] Resumen para {MODEL_NAME} guardado en: {out_json}")

        all_results.append(result)

    # --------------------------
    # 4) Guardar resumen global
    # --------------------------
    global_json = os.path.join(OUT_DIR, "nc_sig_detection_and_defense_all_models.json")
    with open(global_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Resumen global de todos los modelos guardado en: {global_json}")
    print(f"Total de modelos evaluados: {len(all_results)}")


if __name__ == "__main__":
    main()
