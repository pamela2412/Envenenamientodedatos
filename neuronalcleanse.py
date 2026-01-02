import os
import json
import glob
import re
import csv
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ==========================
#   RUTAS Y CONFIG BÁSICA
# ==========================

ROOT_POISON = "Envenenadosbackdoorgota"  # carpeta con modelos .pth envenenados
CLEAN_MODEL_PATH = "GTSRBzip/resnet18_gtsrb_sub18_clean.pth"  # modelo limpio
DATA_H5 = "GTSRBzip/gtsrb-64x64-sub18-clean_train_test.h5"    # dataset limpio (x_train / y_train)
OUT_DIR = "GTSRBzip/neural_cleanse_results"                   # donde dejamos resultados

os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 18
IMG_H = 64
IMG_W = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams de Neural Cleanse (ajusta según GPU)
BATCH_SIZE = 128
NC_STEPS = 800           # iteraciones de optimización por clase
NC_LR = 1e-3             # lr de Adam para mask/pattern
LAMBDA_INIT = 1e-2       # lambda inicial para L1
TARGET_SUCCESS = 0.99    # tasa de éxito deseada del trigger

# Parámetros de fine-tuning de defensa
DEFENSE_EPOCHS = 15        # nº de épocas de fine-tuning
DEFENSE_LR = 1e-4         # learning rate pequeño
DEFENSE_ALPHA = 1.0       # peso de la pérdida sobre imágenes con trigger
DEFENSE_OUT_DIR = "GTSRBzip/neural_cleanse_defended_models"
os.makedirs(DEFENSE_OUT_DIR, exist_ok=True)


# ==========================
#   DATASET LIMPIO H5
# ==========================

class GTSRBH5Dataset(Dataset):
    """
    x en [0,1], y en [0..17]. Aquí NO normalizamos a [-1,1] todavía;
    lo haremos dentro del ataque para poder jugar con patrones en [0,1].
    """
    def __init__(self, x, y):
        # x: (N,H,W,3) float32 en [0,1]
        x = np.asarray(x, dtype=np.float32)
        self.x = torch.from_numpy(x.transpose(0, 3, 1, 2))  # (N,3,H,W)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_clean_data(h5_path=DATA_H5, max_samples=5000):
    """
    Carga el H5 limpio y devuelve un DataLoader con un subconjunto
    (para no matar la GPU).
    """
    with h5py.File(h5_path, "r") as f:
        x_train = np.array(f["x_train"])  # (N,H,W,3) en [0,1]
        y_train = np.array(f["y_train"])

    if max_samples is not None and max_samples < len(x_train):
        idx = np.random.permutation(len(x_train))[:max_samples]
        x_train = x_train[idx]
        y_train = y_train[idx]

    ds = GTSRBH5Dataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
    return dl


# ==========================
#   MODELO (ResNet18)
# ==========================

def build_resnet18(num_classes=NUM_CLASSES):
    """
    Debe ser compatible con el modelo que usaste para entrenar
    'resnet18_gtsrb_sub18_clean.pth'.
    Aquí asumo torchvision ResNet18 con pesos de ImageNet y fc reemplazada.
    """
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==========================
#   NEURAL CLEANSE CORE
# ==========================

@dataclass
class NCResultClass:
    target_class: int
    l1_norm: float
    success_rate: float


@dataclass
class NCResultModel:
    model_name: str
    l1_per_class: list      # list[NCResultClass]
    median_l1: float
    mad: float
    anomaly_indices: list   # por clase
    suspect_classes: list   # clases marcadas como outliers (lado bajo)


def apply_model_norm(x01):
    """
    Convierte imagen en [0,1] -> normalización que espera tu modelo.
    Según tu script original, usaste (x-0.5)*2 para [-1,1].
    """
    return (x01 - 0.5) * 2.0


def optimize_trigger_for_class(model, dataloader, target_class):
    """
    Implementación estilo Neural Cleanse para UNA clase.
    Devuelve máscara, patrón y métricas.
    """
    model.eval()

    # Máscara en [0,1], patrón en [0,1]
    mask = torch.rand(1, 1, IMG_H, IMG_W, device=DEVICE, requires_grad=True)
    pattern = torch.rand(1, 3, IMG_H, IMG_W, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([mask, pattern], lr=NC_LR)
    lambda_reg = LAMBDA_INIT

    data_iter = iter(dataloader)

    for step in range(NC_STEPS):
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            imgs, _ = next(data_iter)

        imgs = imgs.to(DEVICE)  # [0,1]
        bs = imgs.size(0)
        target_labels = torch.full((bs,), target_class, dtype=torch.long, device=DEVICE)

        # constrain mask in [0,1] con sigmoid
        m = torch.sigmoid(mask)     # (1,1,H,W)
        # constrain pattern en [0,1] con sigmoid
        p = torch.sigmoid(pattern)  # (1,3,H,W)

        # broadcast mask a 3 canales
        m3 = m.expand(-1, 3, -1, -1)

        # x_bd en [0,1]
        x_bd = (1 - m3) * imgs + m3 * p

        # normalización a [-1,1] como el modelo
        x_bd_model = apply_model_norm(x_bd)

        logits = model(x_bd_model)
        loss_ce = F.cross_entropy(logits, target_labels)

        # L1 de la máscara (normalizamos por área para comparabilidad)
        l1 = torch.mean(torch.abs(m))

        loss = loss_ce + lambda_reg * l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # medir tasa de éxito
        with torch.no_grad():
            preds = logits.argmax(1)
            success = (preds == target_labels).float().mean().item()

        # ajuste heurístico de λ (como en el paper)
        if (step + 1) % 50 == 0:
            if success > TARGET_SUCCESS:
                lambda_reg *= 1.2
            else:
                lambda_reg /= 1.2
            # evitar valores extremos
            lambda_reg = max(lambda_reg, 1e-6)

        if (step + 1) % 200 == 0:
            print(
                f"  [class {target_class:02d}] step {step+1}/{NC_STEPS} | "
                f"loss_ce={loss_ce.item():.4f} | l1={l1.item():.6f} | succ={success*100:.2f}%"
            )

    # resultado final
    with torch.no_grad():
        m_final = torch.sigmoid(mask).detach().cpu()
        p_final = torch.sigmoid(pattern).detach().cpu()
        l1_final = torch.mean(torch.abs(m_final)).item()

    return m_final, p_final, l1_final, success


def compute_mad_outliers(l1_values):
    """
    Neural Cleanse – MAD based outlier detection.
    l1_values: list[float] (por clase).
    Devuelve mediana, MAD y anomaly_index por clase, más lista de outliers.
    """
    l1_arr = np.array(l1_values)
    median = np.median(l1_arr)
    dev = np.abs(l1_arr - median)
    mad = np.median(dev) + 1e-12  # evitar division por 0

    anomaly = 1.4826 * dev / mad  # factor típico para asimilar a z-score

    # outliers: anomaly > 2 y además por debajo de la mediana (más pequeños)
    suspect = [int(i) for i, (a, val) in enumerate(zip(anomaly, l1_arr)) if a > 2 and val < median]

    return float(median), float(mad), anomaly.tolist(), suspect


def parse_target_from_model_name(model_label: str):
    """
    Extrae el 'tX' del nombre del modelo si existe.
    Ej: 'best_model_gtsrb-64x64-sub18-pois-3-t7-mb21a45-th140.h5' -> 7
    Si no lo encuentra, devuelve None (caso modelo limpio).
    """
    m = re.search(r"-t(\d+)", model_label)
    if not m:
        return None
    return int(m.group(1))


def run_neural_cleanse_for_model(model_path, dataloader, out_dir, model_label=None):
    """
    Ejecuta Neural Cleanse para un modelo (limpio o envenenado).
    model_path: ruta .pth con state_dict (sólo la parte del modelo).
    Devuelve (NCResultModel, detection_summary_dict)
    """
    if model_label is None:
        model_label = os.path.splitext(os.path.basename(model_path))[0]

    print("\n===============================")
    print(f" NEURAL CLEANSE para modelo: {model_label}")
    print("===============================")

    # 1) construir y cargar modelo
    model = build_resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    sd = torch.load(model_path, map_location=DEVICE)

    # por si el dict tiene otras claves (ej: {"netC":..., "optimizerC":...})
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    elif isinstance(sd, dict) and "netC" in sd:
        model.load_state_dict(sd["netC"])
    else:
        model.load_state_dict(sd)

    model.eval()

    class_results = []
    masks = {}
    patterns = {}

    # 2) optimizar trigger para cada clase
    for c in range(NUM_CLASSES):
        print(f"\n>> Optimizando trigger para clase {c} / {NUM_CLASSES-1}")
        m_c, p_c, l1_c, succ_c = optimize_trigger_for_class(
            model, dataloader, target_class=c
        )

        class_results.append(
            NCResultClass(
                target_class=c,
                l1_norm=l1_c,
                success_rate=succ_c,
            )
        )

        masks[c] = m_c  # (1,1,H,W)
        patterns[c] = p_c  # (1,3,H,W)

    # 3) MAD + outliers
    l1_vals = [cr.l1_norm for cr in class_results]
    median_l1, mad, anomaly_indices, suspect_classes = compute_mad_outliers(l1_vals)

    result_model = NCResultModel(
        model_name=model_label,
        l1_per_class=class_results,
        median_l1=median_l1,
        mad=mad,
        anomaly_indices=anomaly_indices,
        suspect_classes=suspect_classes,
    )

    # ========= NUEVO: evaluación de detección =========
    true_target = parse_target_from_model_name(model_label)
    if true_target is not None and len(anomaly_indices) > 0:
        detected_argmax = int(np.argmax(anomaly_indices))
        hit_in_suspect = true_target in suspect_classes
        hit_argmax = detected_argmax == true_target
    else:
        detected_argmax = None
        hit_in_suspect = None
        hit_argmax = None

    detection_summary = {
        "model_name": model_label,
        "true_target": true_target,
        "suspect_classes": suspect_classes,
        "detected_argmax": detected_argmax,
        "hit_in_suspect": hit_in_suspect,
        "hit_argmax": hit_argmax,
        "anomaly_indices": anomaly_indices,
    }

    # ========= 4) guardar resultados =========

    # 4.1 JSON con normas, anomaly index y NUEVOS campos de detección
    out_json = os.path.join(out_dir, f"{model_label}_neural_cleanse.json")
    serializable = {
        "model_name": result_model.model_name,
        "median_l1": result_model.median_l1,
        "mad": result_model.mad,
        "anomaly_indices": result_model.anomaly_indices,
        "suspect_classes": result_model.suspect_classes,
        "per_class": [
            {
                "class": r.target_class,
                "l1_norm": r.l1_norm,
                "success_rate": r.success_rate,
            }
            for r in result_model.l1_per_class
        ],
        # campos extra:
        "true_target": true_target,
        "detected_argmax": detected_argmax,
        "hit_in_suspect": hit_in_suspect,
        "hit_argmax": hit_argmax,
    }
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[NC] Resultados guardados en: {out_json}")

    # 4.2 Guardar tensors de masks/patterns
    out_pt = os.path.join(out_dir, f"{model_label}_neural_cleanse_triggers.pt")
    torch.save({"masks": masks, "patterns": patterns}, out_pt)
    print(f"[NC] Máscaras y patrones guardados en: {out_pt}")

    # 4.3 Guardar PNGs de los 3 triggers con menor L1
    try:
        os.makedirs(os.path.join(out_dir, "triggers_png"), exist_ok=True)
        idx_sorted = np.argsort(l1_vals)  # de menor a mayor
        for rank, c in enumerate(idx_sorted[:3]):
            m_c = masks[c]  # (1,1,H,W)
            p_c = patterns[c]  # (1,3,H,W)

            # Visualizamos el patrón aplicado en toda la imagen (m*pattern)
            m3 = m_c.expand(-1, 3, -1, -1)
            trigger = (m3 * p_c).squeeze(0)  # (3,H,W)
            # trigger está en [0,1]
            out_png = os.path.join(
                out_dir, "triggers_png", f"{model_label}_class{c}_rank{rank}.png"
            )
            torchvision.utils.save_image(trigger, out_png)
        print("[NC] Triggers más compactos guardados como PNG en triggers_png/")
    except Exception as e:
        print(f"[WARN] No se pudieron guardar PNGs de triggers: {e}")

    return result_model, detection_summary


# ==========================
#   FINE-TUNING DE DEFENSA
# ==========================

def fine_tune_with_nc_defense(
    poisoned_model_path: str,
    triggers_pt_path: str,
    suspect_class: int,
    train_loader,
    out_path: str,
    epochs: int = DEFENSE_EPOCHS,
    lr: float = DEFENSE_LR,
    alpha: float = DEFENSE_ALPHA,
):
    """
    Crea un MODELO DEFENDIDO a partir de:
      - un modelo envenenado (poisoned_model_path)
      - la máscara/patrón de Neural Cleanse para la clase suspect_class
      - datos LIMPIOS (train_loader con x en [0,1])

    Guarda los pesos en out_path y devuelve out_path.
    """
    print(f"\n[DEF] Fine-tuning defensa para modelo: {os.path.basename(poisoned_model_path)}")
    print(f"[DEF] Usando clase sospechosa {suspect_class} y triggers de: {triggers_pt_path}")
    print(f"[DEF] Guardando modelo defendido en: {out_path}")

    # 1) Cargar modelo envenenado
    model = build_resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    sd = torch.load(poisoned_model_path, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    elif isinstance(sd, dict) and "netC" in sd:
        model.load_state_dict(sd["netC"])
    else:
        model.load_state_dict(sd)
    model.to(DEVICE)
    model.train()

    # 2) Cargar máscara y patrón de NC
    trig = torch.load(triggers_pt_path, map_location=DEVICE)
    mask = trig["masks"][suspect_class].to(DEVICE)        # (1,1,H,W), [0,1]
    pattern = trig["patterns"][suspect_class].to(DEVICE)  # (1,3,H,W), [0,1]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for ep in range(epochs):
        total_loss = 0.0
        total_steps = 0
        correct_clean = 0
        correct_bd = 0
        total_clean = 0
        total_bd = 0

        for x01, y in train_loader:
            x01 = x01.to(DEVICE)   # [0,1]
            y = y.to(DEVICE)

            bs = x01.size(0)

            # ----- forward limpio -----
            x_clean = apply_model_norm(x01)  # [-1,1]
            logits_clean = model(x_clean)
            loss_clean = criterion(logits_clean, y)

            # ----- forward con trigger de NC -----
            m_batch = mask.expand(bs, 1, IMG_H, IMG_W)
            m_batch = m_batch.expand(bs, 3, IMG_H, IMG_W)   # pasar a 3 canales
            p_batch = pattern.expand(bs, 3, IMG_H, IMG_W)

            x_bd01 = (1 - m_batch) * x01 + m_batch * p_batch      # [0,1]
            x_bd = apply_model_norm(x_bd01)                       # [-1,1]
            logits_bd = model(x_bd)
            loss_bd = criterion(logits_bd, y)      # queremos que mantenga la etiqueta original

            loss = loss_clean + alpha * loss_bd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            with torch.no_grad():
                pred_clean = logits_clean.argmax(1)
                pred_bd = logits_bd.argmax(1)
                correct_clean += (pred_clean == y).sum().item()
                correct_bd += (pred_bd == y).sum().item()
                total_clean += bs
                total_bd += bs

        avg_loss = total_loss / max(total_steps, 1)
        acc_clean = 100.0 * correct_clean / max(total_clean, 1)
        acc_bd = 100.0 * correct_bd / max(total_bd, 1)

        print(
            f"[DEF] Epoch {ep+1}/{epochs} | "
            f"loss={avg_loss:.4f} | acc_clean={acc_clean:.2f}% | acc_bd={acc_bd:.2f}%"
        )

    # 3) Guardar modelo defendido
    torch.save(model.state_dict(), out_path)
    print(f"[DEF] Modelo defendido guardado en: {out_path}")
    return out_path


# ==========================
#   MAIN
# ==========================

def main():
    # 1) Data limpia
    print("Cargando dataset limpio desde H5...")
    dl = load_clean_data(DATA_H5, max_samples=5000)
    print(" Listo.")

    detection_summaries = []

    # 2) Modelo limpio (no se defiende, solo baseline)
    print("\n=== Neural Cleanse en modelo LIMPIO ===")
    _, det_clean = run_neural_cleanse_for_model(
        CLEAN_MODEL_PATH,
        dataloader=dl,
        out_dir=OUT_DIR,
        model_label="clean_resnet18_gtsrb_sub18",
    )
    det_clean["defended_model_path"] = None
    detection_summaries.append(det_clean)

    # 3) Modelos envenenados
    poison_models = sorted(glob.glob(os.path.join(ROOT_POISON, "*.pth")))
    if not poison_models:
        print(f"[WARN] No se encontraron modelos .pth en {ROOT_POISON}")
    else:
        print("\n=== Neural Cleanse en modelos ENVENENADOS + DEFENSA ===")
        for mp in poison_models:
            label = os.path.splitext(os.path.basename(mp))[0]
            _, det_poison = run_neural_cleanse_for_model(
                mp,
                dataloader=dl,
                out_dir=OUT_DIR,
                model_label=label,
            )

            # por defecto no hay modelo defendido
            defended_path = None

            # si encontramos clase sospechosa, aplicamos defensa
            if det_poison["suspect_classes"]:
                suspect = det_poison["suspect_classes"][0]
                trig_pt = os.path.join(OUT_DIR, f"{label}_neural_cleanse_triggers.pt")
                defended_path = os.path.join(DEFENSE_OUT_DIR, f"{label}_nc_defended.pth")

                fine_tune_with_nc_defense(
                    poisoned_model_path=mp,
                    triggers_pt_path=trig_pt,
                    suspect_class=suspect,
                    train_loader=dl,   # usamos el mismo loader limpio
                    out_path=defended_path,
                )
            else:
                print(f"[DEF] {label}: NO se encontró clase sospechosa, no se genera modelo defendido.")

            det_poison["defended_model_path"] = defended_path
            detection_summaries.append(det_poison)

    # 4) RESUMEN GLOBAL (JSON + CSV)
    poisonous = [d for d in detection_summaries if d["true_target"] is not None]
    n_poison = len(poisonous)
    hits_suspect = sum(1 for d in poisonous if d["hit_in_suspect"])
    hits_argmax = sum(1 for d in poisonous if d["hit_argmax"])

    print("\n========== RESUMEN GLOBAL DETECCIÓN ==========")
    print(f"Modelos envenenados evaluados : {n_poison}")
    print(
        f"Detecciones correctas (true_target ∈ suspect_classes): "
        f"{hits_suspect}/{n_poison} "
        f"({(100*hits_suspect/max(n_poison,1)):.1f}%)"
    )
    print(
        f"Detecciones correctas (argmax anomaly == true_target): "
        f"{hits_argmax}/{n_poison} "
        f"({(100*hits_argmax/max(n_poison,1)):.1f}%)"
    )

    # JSON resumen
    summary_json = os.path.join(OUT_DIR, "nc_detection_summary.json")
    with open(summary_json, "w") as f:
        json.dump(detection_summaries, f, indent=2)
    print("Resumen detallado guardado en:", summary_json)

    # CSV resumen
    keys = [
        "model_name",
        "true_target",
        "suspect_classes",
        "detected_argmax",
        "hit_in_suspect",
        "hit_argmax",
        "defended_model_path",
    ]
    summary_csv = os.path.join(OUT_DIR, "nc_detection_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for d in detection_summaries:
            row = {k: d.get(k) for k in keys}
            writer.writerow(row)
    print("Resumen CSV guardado en:", summary_csv)


if __name__ == "__main__":
    main()

