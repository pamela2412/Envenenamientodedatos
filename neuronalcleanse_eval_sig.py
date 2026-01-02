import os
import h5py
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# CONFIGURACIÓN
# ============================================

OUT_DIR = "GTSRBzip"
H5_PATH = os.path.join(OUT_DIR, "gtsrb-64x64-sub18-clean_train_test.h5")

# 🔴 AJUSTAR ESTAS DOS RUTAS:
POISONED_MODEL_PATH = "Modelosenvenenadossig/resnet18_gtsrb_sub18_SIG_f3v4.pth"
DEFENDED_MODEL_PATH = os.path.join(
    OUT_DIR,
    "neural_cleanse_sig_defended_models",
    "resnet18_gtsrb_sub18_SIG_f3v4_sig_defended.pth",
)

# Parámetros ORIGINALES del ataque SIG
TARGET_CLASS = 7      # sig_target_label
SIG_DELTA_EVAL = 40.0 # sig_delta_test
SIG_FREQ_EVAL = 3     # sig_freq

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


# ============================================
# UTILIDADES
# ============================================

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

    img_poison = img + pattern
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


def evaluate_model_sig(
    model,
    x_test,
    y_test,
    target_class,
    delta,
    freq,
    device=DEVICE,
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


# ============================================
# MAIN
# ============================================

def main():
    # 1) Cargar dataset limpio
    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"No se encontró {H5_PATH}")

    print(f"\n📂 Cargando dataset limpio desde: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        x_train = np.array(f["x_train"])
        y_train = np.array(f["y_train"])
        x_test = np.array(f["x_test"])
        y_test = np.array(f["y_test"])
        keep_classes = np.array(f["keep_classes"])

    num_classes = len(np.unique(y_train))
    print(f"Clases remapeadas: {num_classes}")
    print("keep_classes (IDs originales GTSRB en orden remapeado):")
    print(keep_classes.tolist())

    # 2) Cargar modelos
    if not os.path.exists(POISONED_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró modelo envenenado: {POISONED_MODEL_PATH}")
    if not os.path.exists(DEFENDED_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró modelo defendido: {DEFENDED_MODEL_PATH}")

    print("\nCargando modelo envenenado...")
    model_poison = build_resnet18(num_classes=num_classes).to(DEVICE)
    model_poison.load_state_dict(torch.load(POISONED_MODEL_PATH, map_location=DEVICE))

    print("Cargando modelo defendido...")
    model_def = build_resnet18(num_classes=num_classes).to(DEVICE)
    model_def.load_state_dict(torch.load(DEFENDED_MODEL_PATH, map_location=DEVICE))

    # 3) Evaluar con parámetros ORIGINALES del ataque SIG
    print(f"\nEvaluando con parámetros SIG originales: delta={SIG_DELTA_EVAL}, freq={SIG_FREQ_EVAL}, target={TARGET_CLASS}")

    # Modelo envenenado
    acc_clean_orig, asr_orig, cc_o, tot_o, tt_o, tn_o = evaluate_model_sig(
        model_poison,
        x_test,
        y_test,
        target_class=TARGET_CLASS,
        delta=SIG_DELTA_EVAL,
        freq=SIG_FREQ_EVAL,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    # Modelo defendido
    acc_clean_def, asr_def, cc_d, tot_d, tt_d, tn_d = evaluate_model_sig(
        model_def,
        x_test,
        y_test,
        target_class=TARGET_CLASS,
        delta=SIG_DELTA_EVAL,
        freq=SIG_FREQ_EVAL,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    # 4) Imprimir resultados
    print("\n========== RESULTADOS SIG (delta_test, freq originales) ==========")
    print(f"Parámetros SIG eval: delta={SIG_DELTA_EVAL}, freq={SIG_FREQ_EVAL}, target_class={TARGET_CLASS}")
    print(f"Modelo POISONED  - ACC clean: {acc_clean_orig*100:.2f}% ({cc_o}/{tot_o}) "
          f"| ASR_SIG: {asr_orig*100:.2f}% ({tt_o}/{tn_o})")
    print(f"Modelo DEFENDED  - ACC clean: {acc_clean_def*100:.2f}% ({cc_d}/{tot_d}) "
          f"| ASR_SIG: {asr_def*100:.2f}% ({tt_d}/{tn_d})")

    # 5) (Opcional) guardar JSON con resumen
    result = {
        "poisoned_model_path": POISONED_MODEL_PATH,
        "defended_model_path": DEFENDED_MODEL_PATH,
        "num_classes": int(num_classes),
        "target_class": int(TARGET_CLASS),
        "delta_eval": float(SIG_DELTA_EVAL),
        "freq_eval": int(SIG_FREQ_EVAL),
        "acc_clean_poisoned": float(acc_clean_orig),
        "asr_sig_poisoned": float(asr_orig),
        "acc_clean_defended": float(acc_clean_def),
        "asr_sig_defended": float(asr_def),
        "clean_counts_poisoned": [int(cc_o), int(tot_o)],
        "bd_counts_poisoned": [int(tt_o), int(tn_o)],
        "clean_counts_defended": [int(cc_d), int(tot_d)],
        "bd_counts_defended": [int(tt_d), int(tn_d)],
    }

    out_json = os.path.join(OUT_DIR, "nc_sig_eval_delta40_freq3.json")
    import json
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Resumen guardado en: {out_json}")


if __name__ == "__main__":
    main()
