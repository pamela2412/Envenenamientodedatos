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
from torch.utils.data import TensorDataset, DataLoader, Dataset

# ============================================================
# CONFIG GENERAL
# ============================================================

OUT_DIR = "GTSRBzip"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5 = os.path.join(OUT_DIR, "gtsrb-64x64-sub18-clean_train_test.h5")

# Modelo SIG YA ENTRENADO (backdoor)
SIG_MODEL_PATH = os.path.join("Modelosenvenenadossig/resnet18_gtsrb_sub18_SIG_f5.pth")

# Modelo defendido + métricas de defensa
SIG_MODEL_FP_PATH = os.path.join(OUT_DIR, "resnet18_gtsrb_sub18_SIG_f5_finepruned.pth")
SIG_FP_METRICS_JSON = os.path.join(OUT_DIR, "resnet18_gtsrb_sub18_SIG_f5_finepruned_metrics.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ============================================================
# CONFIG SIG (deben coincidir con tu ataque)
# ============================================================

SIG_POISON_RATE = 0.70   # sólo para log (no se usa en defensa)
SIG_DELTA_TRAIN = 35     # no lo usamos acá, pero lo dejo para referencia
SIG_DELTA_TEST = 50      # Δ_ts para test
SIG_FREQ = 5            # frecuencia f
SIG_TARGET_LABEL = 7     # label remapeado de la clase objetivo

# ============================================================
# Utils básicos
# ============================================================

def to_tensor_x(arr):
    """
    arr: numpy (N, H, W, 3) en [0,1]
    -> tensor (N, 3, H, W) en [-1,1]
    """
    t = torch.tensor(arr.transpose(0, 3, 1, 2), dtype=torch.float32)
    t = (t - 0.5) * 2.0
    return t

# ============================================================
# Trigger SIG (igual que tu script, pero solo modo "test")
# ============================================================

def plant_sin_trigger(img, delta=None, f=None):
    """
    Implementación estilo SIG:
      x_b = clip(x + v)
    donde v(i,j) = delta * sin(2π j f / m)

    img: uint8 [H,W,3] en [0,255]
    """
    if delta is None:
        delta = SIG_DELTA_TEST
    if f is None:
        f = SIG_FREQ

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

# ============================================================
# Evaluaciones (ATA / ASR) en silencio
# ============================================================

BATCH_STATS = 128
BATCH_FINETUNE = 128
FT_EPOCHS = 15
LR_FT = 1e-3
WEIGHT_DECAY = 1e-4

PRUNE_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25]
MAX_ACC_DROP = 0.03   # tolerancia de caída de ATA (~3 puntos)

def evaluate_clean_quiet(model, x_test, y_test, device="cuda"):
    model.eval()
    x_test_tensor = to_tensor_x(x_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=BATCH_STATS,
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
    return acc

def evaluate_sig_asr_quiet(model, x_test, y_test,
                           target_label, delta_test, f_test,
                           device="cuda"):
    """
    ASR y accuracy en test envenenado (modo 'test').
    """
    model.eval()

    mask = (y_test != target_label)
    x_other = x_test[mask].copy()
    y_other = y_test[mask]
    n_other = len(y_other)

    if n_other == 0:
        return 0.0, 0.0

    for i in range(n_other):
        img = (x_other[i] * 255.0).astype(np.uint8)
        img_p = plant_sin_trigger(img, delta=delta_test, f=f_test)
        x_other[i] = img_p.astype(np.float32) / 255.0

    x_other_tensor = to_tensor_x(x_other)
    y_other_tensor = torch.tensor(y_other, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(x_other_tensor, y_other_tensor),
        batch_size=BATCH_STATS,
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

            asr_num += (preds == target_label).sum().item()
            asr_den += labels.size(0)

            correct_poisoned += (preds == labels).sum().item()
            total_poisoned += labels.size(0)

    asr = asr_num / max(1, asr_den)
    acc_poisoned = correct_poisoned / max(1, total_poisoned)
    return asr, acc_poisoned

# ============================================================
# Modelo + Fine-Pruning
# ============================================================

class GTSRBDataset(Dataset):
    """Dataset limpio para stats / fine-tuning (normaliza a [-1,1])."""
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]
        t = torch.from_numpy(img).permute(2, 0, 1)
        t = (t - 0.5) * 2.0
        return t, int(self.y[idx])

def build_resnet18_sig(num_classes):
    """Arquitectura base para cargar el modelo SIG desde state_dict."""
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ChannelGate(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("mask", torch.ones(num_channels))

    def forward(self, x):
        return x * self.mask.view(1, -1, 1, 1)

def attach_gate_after_layer4(model: nn.Module, mask: np.ndarray):
    gate = ChannelGate(len(mask))
    gate.mask.data.copy_(torch.from_numpy(mask.astype(np.float32)))
    original_layer4 = model.layer4
    model.layer4 = nn.Sequential(original_layer4, gate)
    return model, gate

def compute_layer4_channel_stats(model: nn.Module,
                                 loader: DataLoader,
                                 max_batches: int = None) -> np.ndarray:
    """
    Media de activación por canal en layer4 (para decidir qué podar).
    """
    model.eval()
    feats_sum = None
    n_samples = 0

    def hook(module, input, output):
        nonlocal feats_sum, n_samples
        with torch.no_grad():
            act = output.detach().abs().mean(dim=(2, 3))  # (B,C)
            if feats_sum is None:
                feats_sum = act.sum(dim=0)
            else:
                feats_sum += act.sum(dim=0)
            n_samples += act.size(0)

    handle = model.layer4.register_forward_hook(hook)

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            _ = model(x)
            if max_batches is not None and (i + 1) >= max_batches:
                break

    handle.remove()
    channel_means = (feats_sum / n_samples).cpu().numpy()
    return channel_means

def build_prune_mask(stats: np.ndarray, prune_ratio: float):
    num_channels = stats.shape[0]
    num_prune = int(prune_ratio * num_channels)
    sorted_idx = np.argsort(stats)  # de menor a mayor activación
    pruned_idx = sorted_idx[:num_prune]
    mask = np.ones(num_channels, dtype=np.float32)
    mask[pruned_idx] = 0.0
    return mask, num_prune

def fine_tune(model: nn.Module, ft_loader: DataLoader,
              epochs: int = FT_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LR_FT, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(ft_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[FT] Epoch {epoch+1}/{epochs} - loss: {running_loss/(i+1):.4f}")

def search_best_prune_ratio_sig(bd_model: nn.Module,
                                val_loader: DataLoader,
                                x_test, y_test,
                                target_label: int,
                                num_classes: int):
    """
    - Calcula stats de layer4.
    - Para cada ratio en PRUNE_RATIOS:
        * crea modelo con gate
        * evalúa ATA y ASR (SIG) sin fine-tuning
    - Devuelve mejor ratio + estadísticas.
    """
    ATA_base = evaluate_clean_quiet(bd_model, x_test, y_test, device=device)
    ASR_base, _ = evaluate_sig_asr_quiet(
        bd_model, x_test, y_test,
        target_label=target_label,
        delta_test=SIG_DELTA_TEST,
        f_test=SIG_FREQ,
        device=device,
    )
    print(f"[Original SIG] ATA: {ATA_base*100:.2f}%, ASR: {ASR_base*100:.2f}%")

    print("[Fine-Pruning SIG] Calculando stats de layer4...")
    stats = compute_layer4_channel_stats(bd_model, val_loader)
    num_channels = stats.shape[0]

    orig_state = bd_model.state_dict()
    candidate_results = []

    for r in PRUNE_RATIOS:
        mask, num_pruned = build_prune_mask(stats, r)

        candidate = build_resnet18_sig(num_classes)
        candidate.load_state_dict(orig_state, strict=True)
        candidate, _ = attach_gate_after_layer4(candidate, mask)
        candidate.to(device)

        ATA_r = evaluate_clean_quiet(candidate, x_test, y_test, device=device)
        ASR_r, _ = evaluate_sig_asr_quiet(
            candidate, x_test, y_test,
            target_label=target_label,
            delta_test=SIG_DELTA_TEST,
            f_test=SIG_FREQ,
            device=device,
        )

        print(
            f"  [ratio={r:.2f}] pruned={num_pruned}/{num_channels}, "
            f"ATA={ATA_r*100:.2f}%, ASR={ASR_r*100:.2f}%"
        )

        candidate_results.append({
            "ratio": r,
            "mask": mask,
            "num_pruned": num_pruned,
            "ATA": ATA_r,
            "ASR": ASR_r,
        })

    viable = [
        c for c in candidate_results
        if c["ATA"] >= ATA_base - MAX_ACC_DROP
    ]

    if viable:
        best = min(viable, key=lambda c: c["ASR"])
        print(
            f"✅ Mejor ratio (con restricción de ATA): {best['ratio']:.2f} "
            f"--> ATA={best['ATA']*100:.2f}%, ASR={best['ASR']*100:.2f}%"
        )
    else:
        best = max(candidate_results, key=lambda c: c["ATA"])
        print(
            "⚠️ Ningún ratio mantiene ATA dentro del umbral; "
            "eligiendo el que menos la destruye:"
        )
        print(
            f"   ratio={best['ratio']:.2f}, "
            f"ATA={best['ATA']*100:.2f}%, ASR={best['ASR']*100:.2f}%"
        )

    return (
        best["ratio"],
        best["mask"],
        best["num_pruned"],
        ATA_base,
        ASR_base,
        best["ATA"],
        best["ASR"],
        num_channels,
    )

def run_fine_pruning_defense_SIG(model_bd,
                                 x_train_clean, y_train_clean,
                                 x_test, y_test,
                                 num_classes):
    """
    Aplica Fine-Pruning adaptativo al modelo SIG:
      - usa train limpio para stats + fine-tuning
      - usa test limpio para ATA y test+trigger para ASR
    """
    n_train = x_train_clean.shape[0]
    n_val = int(0.2 * n_train)

    x_val = x_train_clean[:n_val]
    y_val = y_train_clean[:n_val]
    x_ft = x_train_clean[n_val:]
    y_ft = y_train_clean[n_val:]

    ds_val = GTSRBDataset(x_val, y_val)
    ds_ft = GTSRBDataset(x_ft, y_ft)

    val_loader = DataLoader(
        ds_val, batch_size=BATCH_STATS, shuffle=False,
        num_workers=2, pin_memory=True
    )
    ft_loader = DataLoader(
        ds_ft, batch_size=BATCH_FINETUNE, shuffle=True,
        num_workers=2, pin_memory=True
    )

    (
        prune_ratio_opt,
        mask_opt,
        num_pruned_opt,
        ATA_base,
        ASR_base,
        ATA_podada_sin_ft,
        ASR_podada_sin_ft,
        num_channels,
    ) = search_best_prune_ratio_sig(
        model_bd, val_loader, x_test, y_test,
        target_label=SIG_TARGET_LABEL,
        num_classes=num_classes,
    )

    if ASR_podada_sin_ft >= ASR_base:
        print(
            "⚠️ La mejor poda encontrada no reduce mucho ASR. "
            "Igual hacemos fine-tuning para ver si baja más."
        )

    print(
        f"[Fine-Pruning SIG] Usando prune_ratio_opt={prune_ratio_opt:.2f} "
        f"(pruned {num_pruned_opt}/{num_channels} canales)"
    )

    # Construir modelo podado final desde el modelo original guardado
    fp_model = build_resnet18_sig(num_classes)
    fp_model.load_state_dict(torch.load(SIG_MODEL_PATH, map_location=device))
    fp_model, gate = attach_gate_after_layer4(fp_model, mask_opt)
    fp_model.to(device)

    print("[Fine-Pruning SIG] Fine-tuning del modelo podado con datos limpios...")
    fine_tune(fp_model, ft_loader)

    # Evaluar modelo defendido
    ATA_def = evaluate_clean_quiet(fp_model, x_test, y_test, device=device)
    ASR_def, test_acc_poisoned_def = evaluate_sig_asr_quiet(
        fp_model,
        x_test,
        y_test,
        target_label=SIG_TARGET_LABEL,
        delta_test=SIG_DELTA_TEST,
        f_test=SIG_FREQ,
        device=device,
    )

    print(
        f"[Defendido SIG] ATA_def (clean test): {ATA_def*100:.2f}%, "
        f"ASR_def: {ASR_def*100:.2f}%"
    )

    # Guardar modelo defendido
    torch.save(fp_model.state_dict(), SIG_MODEL_FP_PATH)
    print(f"✅ Modelo SIG defendido guardado en: {SIG_MODEL_FP_PATH}")

    return {
        "prune_ratios_tried": PRUNE_RATIOS,
        "prune_ratio_opt": float(prune_ratio_opt),
        "num_channels_layer4": int(num_channels),
        "num_pruned_opt": int(num_pruned_opt),
        "ATA_original": float(ATA_base),
        "ASR_original": float(ASR_base),
        "ATA_podada_sin_ft": float(ATA_podada_sin_ft),
        "ASR_podada_sin_ft": float(ASR_podada_sin_ft),
        "ATA_def": float(ATA_def),
        "ASR_def": float(ASR_def),
        "test_accuracy_poisoned_def": float(test_acc_poisoned_def),
    }

# ============================================================
# MAIN: solo defensa, SIN reentrenar el ataque
# ============================================================

def main():
    if not os.path.exists(OUT_H5):
        raise FileNotFoundError(
            f"No se encontró {OUT_H5}. Genera primero el H5 limpio."
        )
    if not os.path.exists(SIG_MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo SIG: {SIG_MODEL_PATH}. "
            "Entrénalo primero con tu script de ataque."
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
    print("keep_classes:", keep_classes.tolist())

    # Cargar modelo SIG backdoor ya entrenado
    print(f"📥 Cargando modelo SIG backdoor desde: {SIG_MODEL_PATH}")
    bd_model = build_resnet18_sig(num_classes)
    bd_model.load_state_dict(torch.load(SIG_MODEL_PATH, map_location=device))
    bd_model.to(device)

    # Aplicar defensa Fine-Pruning
    print("\n==============================")
    print("🛡️ Aplicando defensa Fine-Pruning al modelo SIG...")
    print("==============================")

    fp_info = run_fine_pruning_defense_SIG(
        model_bd=bd_model,
        x_train_clean=x_train,
        y_train_clean=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
    )

    # Guardar métricas de defensa
    metrics = {
        "num_classes": int(num_classes),
        "sig_target_label": int(SIG_TARGET_LABEL),
        "sig_poison_rate": float(SIG_POISON_RATE),
        "sig_delta_train": float(SIG_DELTA_TRAIN),
        "sig_delta_test": float(SIG_DELTA_TEST),
        "sig_freq": int(SIG_FREQ),
        "keep_classes_original": [int(x) for x in keep_classes],
        "fine_pruning": fp_info,
    }

    with open(SIG_FP_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Métricas de defensa guardadas en: {SIG_FP_METRICS_JSON}")


if __name__ == "__main__":
    main()
