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

# =========================
# CONFIG GENERAL
# =========================
OUT_DIR = "GTSRBzip"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5 = os.path.join(OUT_DIR, "gtsrb-64x64-sub18-clean_train_test.h5")

# Modelo con backdoor SIG (entrenado desde cero en GTSRB)
SIG_MODEL_PATH = os.path.join(OUT_DIR, "resnet18_gtsrb_sub18_SIG_f5.pth")
SIG_METRICS_JSON = os.path.join(OUT_DIR, "resnet18_gtsrb_sub18_SIG_metrics_f5.json")

BATCH_SIZE = 128
LR = 1e-3
EPOCHS_SIG = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =========================
# CONFIG SIG (globales)
# =========================
# α = 0.7 (porcentaje de la clase target envenenada)
SIG_POISON_RATE = 0.70

# Δ_tr y Δ_ts y frecuencia f del patrón senoidal
SIG_DELTA_TRAIN = 35     # Δ_tr para entrenamiento
SIG_DELTA_TEST = 50      # Δ_ts para test (normalmente más fuerte)
SIG_FREQ = 5         # frecuencia espacial f

# Label remapeado de la clase target (0..17)
SIG_TARGET_LABEL = 7     # AJUSTAR según keep_classes si hace falta

# =========================
# EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=6, delta=1e-4, save_path="best_temp.pth"):
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
# Trigger SIG usando globales
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
# Envenenar train con SIG
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
# Evaluaciones
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
                f"    - sig_from_scratch_example_{k}_before_f{SIG_FREQ}_d{SIG_DELTA_TEST}v4.png"
            )
            print(
                f"    - sig_from_scratch_example_{k}_after_f{SIG_FREQ}_d{SIG_DELTA_TEST}v4.png"
            )

    return asr, acc_poisoned

# =========================
# main
# =========================
def main():
    if not os.path.exists(OUT_H5):
        raise FileNotFoundError(
            f"No se encontró {OUT_H5}. Genera primero el H5 limpio con tu script original."
        )

    print(f"📂 Cargando dataset limpio desde: {OUT_H5}")
    with h5py.File(OUT_H5, "r") as f:
        x_train = np.array(f["x_train"])
        y_train = np.array(f["y_train"])
        x_test = np.array(f["x_test"])
        y_test = np.array(f["y_test"])
        keep_classes = np.array(f["keep_classes"])

    num_classes = len(np.unique(y_train))
    print(f"Clases remapeadas: {num_classes}")
    print("Distribución train por clase:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique.tolist(), counts.tolist())))
    print("keep_classes (IDs originales GTSRB en orden remapeado):")
    print(keep_classes.tolist())

    # Envenenar train con SIG usando globales
    x_train_sig, y_train_sig, poisoned_idx = poison_train_with_sig(
        x_train,
        y_train,
        target_label=SIG_TARGET_LABEL,
        poison_rate=SIG_POISON_RATE,
    )
    print(f"[SIG] Total ejemplos envenenados: {len(poisoned_idx)}")

    # Guardar ejemplo de clase target (limpio y envenenado)
    if len(poisoned_idx) > 0:
        example_idx = int(poisoned_idx[0])

        img_clean = (x_train[example_idx] * 255.0).astype(np.uint8)
        img_clean_bgr = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)

        img_poison = (x_train_sig[example_idx] * 255.0).astype(np.uint8)
        img_poison_bgr = cv2.cvtColor(img_poison, cv2.COLOR_RGB2BGR)

        clean_path = os.path.join(
            OUT_DIR,
            f"sig_target_example_clean_label{SIG_TARGET_LABEL}.png",
        )
        poison_path = os.path.join(
            OUT_DIR,
            f"sig_target_example_poisoned_label{SIG_TARGET_LABEL}_f{SIG_FREQ}_d{SIG_DELTA_TRAIN}.png",
        )

        cv2.imwrite(clean_path, img_clean_bgr)
        cv2.imwrite(poison_path, img_poison_bgr)

        print("[SIG] Ejemplo de clase target guardado:")
        print(f"   - Limpia     -> {clean_path}")
        print(f"   - Envenenada -> {poison_path}")
    else:
        print("[SIG] No se pudo guardar ejemplo target (poisoned_idx vacío).")

    # Tensores y dataloader
    x_train_tensor = to_tensor_x(x_train_sig)
    y_train_tensor = torch.tensor(y_train_sig, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Modelo: ResNet18 preentrenada en ImageNet
    print("🧠 Creando ResNet18 preentrenada en ImageNet (backdoor desde cero en GTSRB)...")
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    early_stopping = EarlyStopping(
        patience=6,
        delta=1e-4,
        save_path=SIG_MODEL_PATH,
    )

    print("🚀 Iniciando entrenamiento con backdoor SIG (desde cero en GTSRB)...")
    for epoch in range(EPOCHS_SIG):
        model.train()
        total_train, correct_train = 0, 0
        epoch_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        epoch_loss /= len(train_loader)
        epoch_acc = correct_train / max(1, total_train)
        print(
            f"[SIG] Epoch {epoch+1:02d} | "
            f"Train Acc: {epoch_acc*100:.2f}% | Loss: {epoch_loss:.4f}"
        )

        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print(f"[SIG] 🛑 Early stopping en epoch {epoch+1}")
            break

    # Cargar mejor modelo
    print(f"[SIG] 📥 Cargando mejor modelo desde: {SIG_MODEL_PATH}")
    model.load_state_dict(torch.load(SIG_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluar limpio
    test_acc = evaluate_clean(model, x_test, y_test, device=device)

    # Evaluar ASR + acc con test envenenado (usa SIG_DELTA_TEST y SIG_FREQ)
    asr_sig, test_acc_poisoned = evaluate_sig_asr(
        model,
        x_test,
        y_test,
        target_label=SIG_TARGET_LABEL,
        device=device,
        save_examples_dir=OUT_DIR,
    )

    metrics = {
        "num_classes": int(num_classes),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
        "test_accuracy_clean": float(test_acc),
        "test_accuracy_poisoned": float(test_acc_poisoned),
        "sig_target_label": int(SIG_TARGET_LABEL),
        "sig_poison_rate": float(SIG_POISON_RATE),
        "sig_delta_train": float(SIG_DELTA_TRAIN),
        "sig_delta_test": float(SIG_DELTA_TEST),
        "sig_freq": int(SIG_FREQ),
        "num_poisoned": int(len(poisoned_idx)),
        "asr_sig": float(asr_sig),
        "keep_classes_original": [int(x) for x in keep_classes],
    }

    with open(SIG_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SIG] Métricas guardadas en: {SIG_METRICS_JSON}")


if __name__ == "__main__":
    main()

