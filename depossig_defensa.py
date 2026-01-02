import os
import json
import h5py
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader

from sklearn.mixture import GaussianMixture

# =========================
# CONFIG GENERAL
# =========================
OUT_DIR = "GTSRBzip"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5 = os.path.join(OUT_DIR, "gtsrb-64x64-sub18-clean_train_test.h5")

# Modelo limpio (mimic)
CLEAN_MODEL_PATH = os.path.join(OUT_DIR, "resnet18_gtsrb_sub18_clean.pth")

# Modelo SIG backdoor ya entrenado (teacher) - mismo que el script de ataque
SIG_MODEL_PATH = os.path.join("Modelosenvenenadossig/resnet18_gtsrb_sub18_SIG_f5.pth")
SIG_MODEL_LABEL = os.path.splitext(os.path.basename(SIG_MODEL_PATH))[0]

# Resultado JSON de la defensa
RESULTS_JSON = os.path.join(
    OUT_DIR, f"{SIG_MODEL_LABEL}_depois_sig_results.json"
)

# Parámetros de entrenamiento defensa
NUM_CLASSES_SUB18 = 18
BATCH_SIZE = 128
LR_DEF = 1e-3
EPOCHS_DEF = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =========================
# CONFIG SIG (globales) - IGUAL QUE EN TU SCRIPT DE ATAQUE
# =========================
# α = 0.7 (porcentaje de la clase target envenenada)
SIG_POISON_RATE = 0.70

# Δ_tr y Δ_ts y frecuencia f del patrón senoidal
SIG_DELTA_TRAIN = 35     # Δ_tr para entrenamiento
SIG_DELTA_TEST = 50      # Δ_ts para test (normalmente más fuerte)
SIG_FREQ = 5          # frecuencia espacial f

# Label remapeado de la clase target (0..17)
SIG_TARGET_LABEL = 7     # AJUSTAR según keep_classes si hace falta


# =========================
# Utils
# =========================
def to_tensor_x(arr):
    """
    arr: numpy (N, H, W, 3) en [0,1]
    -> tensor (N, 3, H, W) en [-1,1]
    """
    t = torch.tensor(arr.transpose(0, 3, 1, 2), dtype=torch.float32)
    t = (t - 0.5) * 2.0
    return t


# =========================
# Trigger SIG usando globales (copiado de tu script)
# =========================
def plant_sin_trigger(img, mode="train"):
    """
    Implementación estilo SIG:
      x_b = clip(x + v)
    donde v(i,j) = delta * sin(2π j f / m)

    Usa variables globales:
      - SIG_FREQ
      - SIG_DELTA_TRAIN si mode="train"
      - SIG_DELTA_TEST  si mode="test"

    img: uint8 [H,W,3] en [0,255]
    """
    img = np.float32(img)
    pattern = np.zeros_like(img, dtype=np.float32)

    H, W, C = pattern.shape
    m = W  # ancho

    if mode == "train":
        delta = SIG_DELTA_TRAIN
    else:
        delta = SIG_DELTA_TEST

    f = SIG_FREQ

    for i in range(H):
        for j in range(W):
            value = delta * np.sin(2 * np.pi * j * f / m)
            # SIN fill de canal vectorizado: bucle explícito en k
            for k in range(C):
                pattern[i, j, k] = value

    # Suma aditiva, sin alpha
    img_poison = img + pattern

    # Clipping a rango válido
    img_poison = np.clip(img_poison, 0, 255).astype(np.uint8)
    return img_poison


# =========================
# Envenenar train con SIG (usa plant_sin_trigger global)
# =========================
def poison_train_with_sig(
    x_train,
    y_train,
    target_label,
    poison_rate=0.2,
):
    """
    Envenena en RAM un porcentaje de la clase target (α = poison_rate)
    usando plant_sin_trigger(..., mode="train") y las globales SIG_*.
    """
    x_poison = x_train.copy()
    idx_target = np.where(y_train == target_label)[0]
    n_target = len(idx_target)
    n_poison = int(n_target * poison_rate)

    if n_poison == 0:
        print("[SIG] Advertencia: n_poison=0 (no se envenena nada)")
        return x_poison, y_train, []

    np.random.shuffle(idx_target)
    poisoned_indices = idx_target[:n_poison]

    print(f"[SIG] Clase objetivo remapeada = {target_label}")
    print(f"[SIG] Total imágenes de esa clase en train = {n_target}")
    print(f"[SIG] Envenenando {n_poison} ({poison_rate*100:.1f}%)")

    for idx in tqdm(poisoned_indices, desc="Poison train"):
        img = (x_poison[idx] * 255.0).astype(np.uint8)
        img_p = plant_sin_trigger(img, mode="train")
        x_poison[idx] = img_p.astype(np.float32) / 255.0

    return x_poison, y_train, poisoned_indices


# =========================
# Evaluaciones (las mismas que en tu script SIG)
# =========================
def evaluate_clean(model, x_test, y_test, device="cuda"):
    """
    Accuracy en test limpio (sin trigger).
    """
    model.eval()
    x_test_tensor = to_tensor_x(x_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / max(1, total)
    print(f"[SIG] Accuracy test limpia: {acc*100:.2f}% ({correct}/{total})")
    return acc


def evaluate_sig_asr(
    model,
    x_test,
    y_test,
    target_label,
    device="cuda",
    save_examples_dir=None,
):
    """
    Calcula:
      - ASR (Attack Success Rate) usando plant_sin_trigger(..., mode="test")
      - Accuracy sobre test envenenado
      - Guarda ejemplos antes/después si save_examples_dir != None
    """
    model.eval()

    # Usamos sólo ejemplos cuya etiqueta != target_label
    mask = (y_test != target_label)
    x_other = x_test[mask].copy()
    y_other = y_test[mask]
    n_other = len(y_other)

    if n_other == 0:
        print("[SIG] No hay ejemplos != target en test.")
        return 0.0, 0.0

    print(f"[SIG] Aplicando trigger SIG a {n_other} imágenes de test...")

    # Guardar copias de los primeros 3 ejemplos ANTES de envenenar
    num_save = min(3, n_other)
    orig_examples = (x_other[:num_save] * 255.0).astype(np.uint8)

    # Aplicar trigger SIG (Δ_ts, SIG_DELTA_TEST) a TODO x_other
    for i in tqdm(range(n_other), desc="SIG test"):
        img = (x_other[i] * 255.0).astype(np.uint8)
        img_p = plant_sin_trigger(img, mode="test")
        x_other[i] = img_p.astype(np.float32) / 255.0

    # Copias de los primeros 3 ejemplos DESPUÉS de envenenar
    poisoned_examples = (x_other[:num_save] * 255.0).astype(np.uint8)

    x_other_tensor = to_tensor_x(x_other)
    y_other_tensor = torch.tensor(y_other, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(x_other_tensor, y_other_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    asr_num, asr_den = 0, 0
    correct_poisoned, total_poisoned = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            # ASR: cuántas se van a la clase objetivo
            asr_num += (preds == target_label).sum().item()
            asr_den += labels.size(0)

            # Accuracy test envenenado (pred vs etiqueta real)
            correct_poisoned += (preds == labels).sum().item()
            total_poisoned += labels.size(0)

    asr = asr_num / max(1, asr_den)
    acc_poisoned = correct_poisoned / max(1, total_poisoned)

    print(
        f"[SIG] ASR: {asr*100:.2f}% ({asr_num}/{asr_den}) "
        f"con delta_test={SIG_DELTA_TEST}, f={SIG_FREQ}"
    )
    print(
        f"[SIG] Accuracy test ENVENENADO: {acc_poisoned*100:.2f}% "
        f"({correct_poisoned}/{total_poisoned})"
    )

    # Guardar ejemplos antes/después
    if save_examples_dir is not None and num_save > 0:
        os.makedirs(save_examples_dir, exist_ok=True)
        for k in range(num_save):
            before_path = os.path.join(
                save_examples_dir,
                f"sig_from_scratch_example_{k}_before_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png",
            )
            after_path = os.path.join(
                save_examples_dir,
                f"sig_from_scratch_example_{k}_after_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png",
            )

            img_before = cv2.cvtColor(orig_examples[k], cv2.COLOR_RGB2BGR)
            img_after = cv2.cvtColor(poisoned_examples[k], cv2.COLOR_RGB2BGR)

            cv2.imwrite(before_path, img_before)
            cv2.imwrite(after_path, img_after)

        print(f"[SIG] Ejemplos antes/después guardados en {save_examples_dir}:")
        for k in range(num_save):
            print(
                f"    - sig_from_scratch_example_{k}_before_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png"
            )
            print(
                f"    - sig_from_scratch_example_{k}_after_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png"
            )

    return asr, acc_poisoned


# =========================
# Modelos (clean y defended)
# =========================
def build_resnet18(num_classes):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_clean_model(num_classes):
    """
    Carga el modelo limpio desde CLEAN_MODEL_PATH.
    """
    model = build_resnet18(num_classes)
    sd = torch.load(CLEAN_MODEL_PATH, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("✅ Modelo limpio cargado desde:", CLEAN_MODEL_PATH)
    return model


def load_sig_model(num_classes):
    """
    Carga el modelo SIG backdoor entrenado (teacher) desde SIG_MODEL_PATH.
    """
    model = build_resnet18(num_classes)
    sd = torch.load(SIG_MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    print("✅ Modelo SIG (backdoor) cargado desde:", SIG_MODEL_PATH)
    return model


class EarlyStopping:
    def __init__(self, patience=6, delta=1e-4, save_path="best_defended_temp.pth"):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, train_loss, model):
        if train_loss < self.best_loss - self.delta:
            self.best_loss = train_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_defended_model(x_train_def_tensor, y_train_def, num_classes, save_path):
    """
    Entrena el modelo defendido desde cero sobre el train 'limpio estimado'
    (muestras que NO fueron marcadas como poison).
    """
    train_loader_def = DataLoader(
        TensorDataset(x_train_def_tensor, torch.tensor(y_train_def, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model_def = build_resnet18(num_classes).to(device)
    criterion_def = nn.CrossEntropyLoss()
    optimizer_def = optim.Adam(model_def.parameters(), lr=LR_DEF)
    early_stopping_def = EarlyStopping(patience=6, save_path=save_path)

    print("Iniciando entrenamiento DEFENDIDO (De-Pois-lite SIG)...")
    for epoch in range(EPOCHS_DEF):
        model_def.train()
        total_train_def, correct_train_def, epoch_loss_def = 0, 0, 0.0

        for imgs, labels in train_loader_def:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_def.zero_grad()
            outputs = model_def(imgs)
            loss = criterion_def(outputs, labels)
            loss.backward()
            optimizer_def.step()

            epoch_loss_def += loss.item()
            _, preds = torch.max(outputs, 1)
            total_train_def += labels.size(0)
            correct_train_def += (preds == labels).sum().item()

        epoch_loss_def /= len(train_loader_def)
        epoch_acc_def = correct_train_def / total_train_def
        print(f"[DEF] E {epoch+1:02d} | Acc: {epoch_acc_def*100:.2f}% | Loss: {epoch_loss_def:.4f}")

        early_stopping_def(epoch_loss_def, model_def)
        if early_stopping_def.early_stop:
            print(f"🛑 [DEF] Early stopping en epoch {epoch+1}")
            break

    model_def.load_state_dict(torch.load(save_path, map_location=device))
    model_def = model_def.to(device)
    return model_def


# =========================
# Scoring De-Pois-lite y GMM unknown-κ
# =========================
def score_poison_with_clean_model(x_train_poison_tensor, y_train_poison,
                                  teacher_model, clean_model):
    """
    Devuelve:
      - scores: np.array de tamaño N_train con ||softmax_T - softmax_clean||_1
    """
    teacher_model.eval()
    clean_model.eval()

    dataset = TensorDataset(x_train_poison_tensor,
                            torch.tensor(y_train_poison, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    scores = np.zeros(len(x_train_poison_tensor), dtype=np.float32)
    start = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="De-Pois scoring (SIG)"):
            imgs = imgs.to(device)

            logits_teacher = teacher_model(imgs)
            logits_clean = clean_model(imgs)

            p_t = torch.softmax(logits_teacher, dim=1)
            p_c = torch.softmax(logits_clean, dim=1)

            batch_scores = torch.sum(torch.abs(p_t - p_c), dim=1).cpu().numpy()

            end = start + len(batch_scores)
            scores[start:end] = batch_scores
            start = end

    return scores


def select_poison_gmm(scores,
                      min_fraction=0.01,
                      max_fraction=0.4,
                      prob_thresh=0.8,
                      random_state=0):
    """
    unknown-κ con GMM 1D
    """
    N = len(scores)
    scores_2d = scores.reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=random_state
        )
        gmm.fit(scores_2d)
        means = gmm.means_.reshape(-1)
        poison_comp = np.argmax(means)          # componente con media más alta
        probs = gmm.predict_proba(scores_2d)[:, poison_comp]
    except Exception as e:
        print("⚠️ GMM falló, usando fallback top-5%:", e)
        probs = np.zeros(N, dtype=np.float32)
        order = np.argsort(scores)[::-1]
        k = max(1, int(0.05 * N))
        probs[order[:k]] = 1.0

    mask_poison = probs > prob_thresh
    n_poison = mask_poison.sum()

    min_count = max(1, int(round(min_fraction * N)))
    max_count = int(round(max_fraction * N))

    if n_poison < min_count:
        top_idx = np.argsort(probs)[::-1][:min_count]
        mask_poison[:] = False
        mask_poison[top_idx] = True
        n_poison = min_count
    elif n_poison > max_count:
        candidate_idx = np.where(mask_poison)[0]
        candidate_probs = probs[candidate_idx]
        order = np.argsort(candidate_probs)[::-1][:max_count]
        final_idx = candidate_idx[order]
        mask_poison[:] = False
        mask_poison[final_idx] = True
        n_poison = max_count

    idx_poison = np.where(mask_poison)[0]
    frac_est = n_poison / N
    return idx_poison, probs, frac_est


# =========================
# MAIN
# =========================
def main():
    # 1) Cargar dataset limpio
    if not os.path.exists(OUT_H5):
        raise FileNotFoundError(
            f"No se encontró {OUT_H5}. Genera primero el H5 limpio."
        )

    print(f"📂 Cargando dataset limpio desde: {OUT_H5}")
    with h5py.File(OUT_H5, "r") as f:
        x_train = np.array(f["x_train"])   # [0,1]
        y_train = np.array(f["y_train"])
        x_test = np.array(f["x_test"])
        y_test = np.array(f["y_test"])

    num_classes = len(np.unique(y_train))
    print(f"Clases remapeadas: {num_classes}")

    # 2) Reconstruir train envenenado SIG en RAM usando las MISMAS globales
    x_train_sig, y_train_sig, poisoned_idx = poison_train_with_sig(
        x_train,
        y_train,
        target_label=SIG_TARGET_LABEL,
        poison_rate=SIG_POISON_RATE,
    )
    print(f"[SIG] Total ejemplos envenenados (reconstruidos): {len(poisoned_idx)}")

    # Tensores para scoring y defensa
    x_train_sig_tensor = to_tensor_x(x_train_sig)
    y_train_sig_tensor = torch.tensor(y_train_sig, dtype=torch.long)

    # 3) Cargar modelos: teacher (SIG) y mimic (clean)
    teacher_model = load_sig_model(NUM_CLASSES_SUB18)
    clean_model = load_clean_model(NUM_CLASSES_SUB18)

    # 4) Métricas BEFORE (modelo SIG original) usando las funciones globales
    print("\n=== BEFORE (modelo SIG original) ===")
    acc_clean_before = evaluate_clean(teacher_model, x_test, y_test, device=device)
    asr_before, acc_poisoned_before = evaluate_sig_asr(
        teacher_model,
        x_test,
        y_test,
        target_label=SIG_TARGET_LABEL,
        device=device,
        save_examples_dir=None,  # si querés imágenes, pon OUT_DIR
    )

    # 5) Scoring De-Pois-lite con modelo limpio
    scores = score_poison_with_clean_model(
        x_train_sig_tensor, y_train_sig,
        teacher_model, clean_model
    )

    # 6) Selección unknown-κ (GMM)
    idx_poison_pred, probs_poison, frac_est = select_poison_gmm(
        scores,
        min_fraction=0.01,
        max_fraction=0.6,
        prob_thresh=0.6,
        random_state=0,
    )

    N = len(scores)
    mask_keep = np.ones(N, dtype=bool)
    mask_keep[idx_poison_pred] = False
    num_removed = (~mask_keep).sum()
    print(f"De-Pois-lite (SIG) marcó {num_removed} muestras como poison "
          f"({num_removed/N:.2%} del train, κ_est ≈ {frac_est*100:.2f}%).")

    # 7) Re-entrenar modelo DEFENDIDO sobre las muestras keep
    x_train_def_tensor = x_train_sig_tensor[mask_keep]
    y_train_def = y_train_sig[mask_keep]

    def_ckpt = os.path.join(OUT_DIR, f"{SIG_MODEL_LABEL}_depois_sig_defended_best.pth")
    defended_model = train_defended_model(
        x_train_def_tensor, y_train_def,
        NUM_CLASSES_SUB18, def_ckpt
    )

    # 8) Métricas AFTER (usando las mismas funciones)
    print("\n=== AFTER (modelo DEFENDIDO De-Pois SIG) ===")
    acc_clean_after = evaluate_clean(defended_model, x_test, y_test, device=device)
    asr_after, acc_poisoned_after = evaluate_sig_asr(
        defended_model,
        x_test,
        y_test,
        target_label=SIG_TARGET_LABEL,
        device=device,
        save_examples_dir=None,
    )

    # 9) Guardar resultados
    results = {
        "model_name": SIG_MODEL_LABEL,
        "sig_target_label": int(SIG_TARGET_LABEL),
        "sig_poison_rate": float(SIG_POISON_RATE),
        "sig_delta_train": float(SIG_DELTA_TRAIN),
        "sig_delta_test": float(SIG_DELTA_TEST),
        "sig_freq": int(SIG_FREQ),

        "num_train": int(N),
        "num_poison_reconstructed": int(len(poisoned_idx)),

        "num_removed_depois": int(num_removed),
        "kappa_estimated": float(frac_est),

        "test_accuracy_clean_before": float(acc_clean_before),
        "test_accuracy_poisoned_before": float(acc_poisoned_before),
        "asr_sig_before": float(asr_before),

        "test_accuracy_clean_after": float(acc_clean_after),
        "test_accuracy_poisoned_after": float(acc_poisoned_after),
        "asr_sig_after": float(asr_after),
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados De-Pois-lite SIG guardados en: {RESULTS_JSON}")


if __name__ == "__main__":
    main()

