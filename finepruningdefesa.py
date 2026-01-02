# ============================================================
# 🛡️ Fine-Pruning + ASR adaptativo para ataque de lluvia en GTSRB
#
# Recorre todos los H5 envenenados de una carpeta, empareja el modelo .pth
# correspondiente, y para cada par:
#   - Calcula ATA y ASR con el modelo backdoor original
#   - Busca el mejor ratio de poda (PRUNE_RATIOS) que baje ASR sin matar ATA
#   - Aplica Fine-Pruning (poda en layer4 + fine-tuning limpio)
#   - Calcula ATA_def y ASR_def con el modelo defendido
#   - Guarda el modelo defendido y loguea todo en un CSV
#
# Requiere:
#   - Dataset limpio:  gtsrb-64x64-sub18-clean_train_test.h5
#   - Datasets poison: gtsrb-sub18-pois-*-t*-*.h5
#   - CSV de rows poison: gtsrb-sub18-pois-..._poisoned_rows.csv (opcional)
#   - Modelos envenenados .pth (ResNet18 con 18 clases)
# ============================================================

import os
import re
import glob
import random

import h5py
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

# ---------------------------
# ⚙️ Configuración general
# ---------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 18
IMG_HEIGHT = 64
IMG_WIDTH = 64

# H5 limpio (no envenenado)
CLEAN_H5 = "GTSRBzip/gtsrb-64x64-sub18-clean_train_test.h5"

# Carpeta de H5 envenenados + CSV
POISON_H5_DIR = "GTSRBzip/poisoned_from_torchvision_64x64_train_test"

# Carpeta con modelos envenenados
MODELS_DIR = "Modelosenvenenadosgotas"   # 👈 AJUSTA ESTA RUTA

# Prefijo/sufijo para construir nombre del modelo a partir del H5
# Ejemplo: H5 = gtsrb-sub18-pois-3-t7-mb21a45-th140.h5
#          Modelo = resnet18_gtsrb-sub18-pois-3-t7-mb21a45-th140.pth
MODEL_PREFIX = "best_model_"
MODEL_SUFFIX = ".h5.pth"

# Carpeta salida para modelos defendidos + resultados
OUT_DIR = os.path.join(MODELS_DIR, "finepruned_with_asr")
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUT_DIR, "finepruning_with_asr_results.csv")

# Hiperparámetros Fine-Pruning (adaptativos)
PRUNE_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25]  # candidatos (5% a 25%)
MAX_ACC_DROP = 0.03   # caída máxima tolerada en accuracy limpia (~3 puntos)

BATCH_STATS = 128       # batch para medir activaciones y test
BATCH_FINETUNE = 128    # batch para fine-tuning
FT_EPOCHS = 15
FT_LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2      # parte de train limpio para medir activaciones

# Parámetros del trigger de lluvia (los mismos que usaste al envenenar)
K1_SIZE_DEFAULT = 5
REFRACTION_INTENSITY = 0.04
SPECULAR_INTENSITY = 25
BLEND_ALPHA = 0.5

# Para reproducir siempre el mismo patrón de lluvia
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1. Trigger de lluvia (mismas funciones de tu ataque)
# ============================================================

def create_rain_trigger_advanced(
        shape,
        k1_size=5,
        k2_size=21,
        mask_threshold=150,
        refraction_intensity=0.04,
        use_motion_blur=False,
        motion_blur_strength=15,
        motion_blur_angle=45
):
    height, width = shape
    noise = np.random.uniform(0, 255, (height, width))
    blur1 = cv2.blur(noise, (k1_size, k1_size))

    if use_motion_blur:
        kernel_size = motion_blur_strength
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        angle_rad = motion_blur_angle * np.pi / 180.0
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)

        if np.abs(dx) > np.abs(dy):
            steps = int(np.abs(dx * center))
            for i in range(-steps, steps + 1):
                y = int(i * (dy / dx))
                if 0 <= center + i < kernel_size and 0 <= center + y < kernel_size:
                    kernel[center + y, center + i] = 1
        else:
            steps = int(np.abs(dy * center))
            for i in range(-steps, steps + 1):
                x = int(i * (dx / dy))
                if 0 <= center + x < kernel_size and 0 <= center + i < kernel_size:
                    kernel[center + i, center + x] = 1

        if np.sum(kernel) == 0:
            kernel[center, center] = 1
        kernel = kernel / np.sum(kernel)
        motion_blur = cv2.filter2D(blur1, -1, kernel)
    else:
        motion_blur = cv2.GaussianBlur(blur1, (k2_size, k2_size), 0)

    raindrop_pattern = cv2.normalize(
        motion_blur, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, mask_binary = cv2.threshold(
        raindrop_pattern, mask_threshold, 255, cv2.THRESH_BINARY
    )
    mask_norm = mask_binary.astype(np.float32) / 255.0

    gx = cv2.Sobel(raindrop_pattern, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(raindrop_pattern, cv2.CV_32F, 0, 1, ksize=3)
    disp_x = gx * refraction_intensity
    disp_y = gy * refraction_intensity

    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (map_x + disp_x).astype(np.float32)
    map_y = (map_y + disp_y).astype(np.float32)

    specular_map = cv2.GaussianBlur(mask_binary, (19, 19), sigmaX=6)
    specular_map = cv2.normalize(specular_map, None, 0, 1, cv2.NORM_MINMAX)

    mask_rgb = np.stack([mask_norm] * 3, axis=-1)
    specular_rgb = np.stack([specular_map] * 3, axis=-1)

    return {
        "mask_rgb": mask_rgb,
        "map_x": map_x,
        "map_y": map_y,
        "specular_rgb": specular_rgb,
    }


def apply_trigger_with_advanced_effects(
        image_original_bgr,
        trigger_components,
        specular_intensity=25,
        blend_alpha=0.25
):
    image_f = image_original_bgr.astype(np.float32)
    map_x = trigger_components["map_x"]
    map_y = trigger_components["map_y"]
    mask_rgb = trigger_components["mask_rgb"]
    specular_rgb = trigger_components["specular_rgb"]

    refracted_image = cv2.remap(
        image_f, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    reflected_with_specular = refracted_image + specular_rgb * specular_intensity

    blended = (
            image_f * (1 - blend_alpha * mask_rgb)
            + reflected_with_specular * (blend_alpha * mask_rgb)
    )

    out = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def apply_trigger_batch_tensor(x_tensor, trigger_components):
    """
    x_tensor: (B, 3, H, W) en [0,1] (float32)
    Devuelve tensor (B, 3, H, W) con trigger aplicado, también en [0,1].
    """
    x_np = x_tensor.cpu().numpy()  # (B, 3, H, W)
    out_list = []

    for img_chw in x_np:
        img_rgb = np.transpose(img_chw, (1, 2, 0)) * 255.0
        img_u8 = img_rgb.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        poisoned_bgr = apply_trigger_with_advanced_effects(
            img_bgr,
            trigger_components,
            SPECULAR_INTENSITY,
            BLEND_ALPHA,
        )

        poisoned_rgb = cv2.cvtColor(poisoned_bgr, cv2.COLOR_BGR2RGB)
        poisoned = poisoned_rgb.astype(np.float32) / 255.0
        poisoned_chw = np.transpose(poisoned, (2, 0, 1))
        out_list.append(poisoned_chw)

    out_np = np.stack(out_list, axis=0)
    return torch.from_numpy(out_np)


# ============================================================
# 2. Dataset limpio desde H5
# ============================================================

class H5ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]  # (H, W, 3), float32 en [0,1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        label = int(self.y[idx])
        return img_t, label


def load_clean_data(h5_path: str, val_fraction: float = VAL_FRACTION):
    with h5py.File(h5_path, "r") as f:
        x_train = f["x_train"][:]
        y_train = f["y_train"][:]
        x_test = f["x_test"][:]
        y_test = f["y_test"][:]

    n_train = x_train.shape[0]
    n_val = int(n_train * val_fraction)

    x_val = x_train[:n_val]
    y_val = y_train[:n_val]
    x_ft = x_train[n_val:]
    y_ft = y_train[n_val:]

    ds_val = H5ArrayDataset(x_val, y_val)
    ds_ft = H5ArrayDataset(x_ft, y_ft)
    ds_test = H5ArrayDataset(x_test, y_test)

    val_loader = DataLoader(ds_val, batch_size=BATCH_STATS, shuffle=False,
                            num_workers=2, pin_memory=True)
    ft_loader = DataLoader(ds_ft, batch_size=BATCH_FINETUNE, shuffle=True,
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=BATCH_STATS, shuffle=False,
                             num_workers=2, pin_memory=True)

    return val_loader, ft_loader, test_loader


# ============================================================
# 3. Modelo + Fine-Pruning
# ============================================================

def build_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_backdoor_model(model_path: str) -> nn.Module:
    model = build_resnet18()
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(DEVICE)
    return model


def compute_layer4_channel_stats(model: nn.Module,
                                 loader: DataLoader,
                                 max_batches: int = None) -> np.ndarray:
    model.eval()
    feats_sum = None
    n_samples = 0

    def hook(module, input, output):
        nonlocal feats_sum, n_samples
        # output: (B, C, H, W)
        with torch.no_grad():
            act = output.detach().abs().mean(dim=(2, 3))  # (B, C)
            if feats_sum is None:
                feats_sum = act.sum(dim=0)
            else:
                feats_sum += act.sum(dim=0)
            n_samples += act.size(0)

    handle = model.layer4.register_forward_hook(hook)

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(DEVICE, non_blocking=True)
            _ = model(x)
            if max_batches is not None and (i + 1) >= max_batches:
                break

    handle.remove()
    channel_means = (feats_sum / n_samples).cpu().numpy()
    return channel_means


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


def build_prune_mask(stats: np.ndarray, prune_ratio: float):
    num_channels = stats.shape[0]
    num_prune = int(prune_ratio * num_channels)
    sorted_idx = np.argsort(stats)  # de menor a mayor
    pruned_idx = sorted_idx[:num_prune]
    mask = np.ones(num_channels, dtype=np.float32)
    mask[pruned_idx] = 0.0
    return mask, num_prune


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_asr(model: nn.Module,
                 loader: DataLoader,
                 target_label: int,
                 trigger_components) -> float:
    """
    ASR = P(model(x_triggered) == target | y != target)
    usando test limpio + trigger de lluvia.
    """
    model.eval()
    success = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            mask = (y != target_label)
            if mask.sum().item() == 0:
                continue

            x_nt = x[mask]
            y_nt = y[mask]

            x_trig = apply_trigger_batch_tensor(x_nt, trigger_components)
            x_trig = x_trig.to(DEVICE, non_blocking=True)

            logits = model(x_trig)
            preds = logits.argmax(dim=1)

            success += (preds == target_label).sum().item()
            total += y_nt.size(0)

    return success / total if total > 0 else 0.0


def fine_tune(model: nn.Module, ft_loader: DataLoader,
              epochs: int = FT_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(ft_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[FT] Epoch {epoch + 1}/{epochs} - loss: {running_loss / (i + 1):.4f}")


# ============================================================
# 4. Parsear nombre del H5 para sacar parámetros de trigger
# ============================================================

def parse_poison_filename(h5_path: str):
    """
    Espera algo tipo:
      gtsrb-sub18-pois-3-t7-mb21a45-th140.h5
      gtsrb-sub18-pois-10-t7-k221-th160.h5
    Devuelve dict con:
      - poison_rate (en %)
      - target (int)
      - blur_type (string completo, p.ej. 'mb21a45', 'k221')
      - threshold (int)
    """
    base = os.path.basename(h5_path)
    if base.endswith(".h5"):
        base = base[:-3]

    pattern = r"gtsrb-sub\d+-pois-(\d+)-t(\d+)-([^-]+)-th(\d+)"
    m = re.match(pattern, base)
    if not m:
        raise ValueError(f"No pude parsear el nombre del H5: {base}")

    pois_pct = int(m.group(1))
    target = int(m.group(2))
    blur_type = m.group(3)
    threshold = int(m.group(4))

    return {
        "poison_rate": pois_pct,
        "target": target,
        "blur_type": blur_type,
        "threshold": threshold,
        "base_name": base,
    }


def build_trigger_from_info(info):
    blur_type = info["blur_type"]
    threshold = info["threshold"]

    use_motion_blur = blur_type.startswith("mb")
    k1_size = K1_SIZE_DEFAULT
    k2_size = 21
    motion_blur_strength = 15
    motion_blur_angle = 45

    if use_motion_blur:
        # Ejemplo: 'mb21a45'
        m = re.match(r"mb(\d+)(?:a(\d+))?", blur_type)
        if m:
            motion_blur_strength = int(m.group(1))
            if m.group(2) is not None:
                motion_blur_angle = int(m.group(2))
        k2_size = 21  # no se usa realmente en motion blur
    else:
        # Ejemplo: 'k221'
        m = re.match(r"k2(\d+)", blur_type)
        if m:
            k2_size = int(m.group(1))
        use_motion_blur = False

    trigger_components = create_rain_trigger_advanced(
        (IMG_HEIGHT, IMG_WIDTH),
        k1_size=k1_size,
        k2_size=k2_size,
        mask_threshold=threshold,
        refraction_intensity=REFRACTION_INTENSITY,
        use_motion_blur=use_motion_blur,
        motion_blur_strength=motion_blur_strength,
        motion_blur_angle=motion_blur_angle,
    )

    return trigger_components


# ============================================================
# 5. Búsqueda adaptativa del mejor ratio de poda
# ============================================================

def search_best_prune_ratio(bd_model: nn.Module,
                            val_loader: DataLoader,
                            test_loader: DataLoader,
                            trigger_components,
                            target_label: int):
    """
    Busca el mejor PRUNE_RATIO para este modelo:
    - Calcula stats de layer4.
    - Para cada ratio candidato:
        - Crea modelo con gate y esa máscara.
        - Evalúa ATA y ASR sin fine-tuning.
    - Devuelve:
        - prune_ratio_opt, mask_opt, num_pruned_opt,
        - ATA_base, ASR_base,
        - ATA_opt_sin_ft, ASR_opt_sin_ft,
        - num_channels
    """
    # Accuracy y ASR del modelo original (sin poda)
    ATA_base = evaluate_accuracy(bd_model, test_loader)
    ASR_base = evaluate_asr(bd_model, test_loader, target_label, trigger_components)

    print(f"[Original] ATA: {ATA_base*100:.2f}%, ASR: {ASR_base*100:.2f}%")

    # 1) Stats de layer4 con datos limpios
    print("Calculando stats de layer4 para búsqueda de poda...")
    stats = compute_layer4_channel_stats(bd_model, val_loader)
    num_channels = stats.shape[0]

    # Guardar el estado original para clonarlo en cada prueba
    orig_state = bd_model.state_dict()

    candidate_results = []

    for r in PRUNE_RATIOS:
        mask, num_pruned = build_prune_mask(stats, r)

        candidate = build_resnet18()
        candidate.load_state_dict(orig_state, strict=True)
        candidate, _ = attach_gate_after_layer4(candidate, mask)
        candidate.to(DEVICE)

        ATA_r = evaluate_accuracy(candidate, test_loader)
        ASR_r = evaluate_asr(candidate, test_loader, target_label, trigger_components)

        print(
            f"  [ratio={r:.2f}] "
            f"pruned={num_pruned}/{num_channels}, "
            f"ATA={ATA_r*100:.2f}%, ASR={ASR_r*100:.2f}%"
        )

        candidate_results.append({
            "ratio": r,
            "mask": mask,
            "num_pruned": num_pruned,
            "ATA": ATA_r,
            "ASR": ASR_r,
        })

    # 2) Seleccionamos el mejor ratio con restricción de accuracy
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
        # Si ninguno cumple el criterio, elegimos el de mayor ATA
        best = max(candidate_results, key=lambda c: c["ATA"])
        print(
            "⚠️ Ningún ratio mantiene la ATA dentro del umbral, "
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


# ============================================================
# 6. Fine-Pruning + ASR para un par (modelo, H5 poison)
# ============================================================

def run_fine_pruning_with_asr_for_pair(model_path: str,
                                       h5_poison_path: str,
                                       val_loader: DataLoader,
                                       ft_loader: DataLoader,
                                       test_loader: DataLoader):
    info = parse_poison_filename(h5_poison_path)
    base_name = info["base_name"]
    target = info["target"]

    print(f"\n====== Procesando ataque: {base_name} ======")
    print(f"Modelo: {model_path}")
    print(f"Target: {target}, poison_rate: {info['poison_rate']}%")

    # CSV con rows envenenados (opcional)
    csv_path = os.path.join(
        POISON_H5_DIR,
        base_name + "_poisoned_rows.csv"
    )
    csv_poison_count = None
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        csv_poison_count = len(df_csv)
        print(f"CSV encontrado: {csv_path}, filas poison: {csv_poison_count}")
    else:
        print("⚠️ CSV de rows poison NO encontrado (no pasa nada para ASR).")
        csv_poison_count = -1

    # Para log: número de samples de train en este H5 poison
    with h5py.File(h5_poison_path, "r") as f:
        num_train = f["x_train"].shape[0]

    # 1) Construir trigger de lluvia específico para este H5
    trigger_components = build_trigger_from_info(info)

    # 2) Modelo backdoor original (para búsqueda)
    bd_model = load_backdoor_model(model_path)

    # 3) Buscar mejor ratio de poda para ESTE modelo
    (
        prune_ratio_opt,
        mask_opt,
        num_pruned_opt,
        ATA_base,
        ASR_base,
        ATA_opt_sin_ft,
        ASR_opt_sin_ft,
        num_channels,
    ) = search_best_prune_ratio(
        bd_model, val_loader, test_loader, trigger_components, target
    )

    if ASR_opt_sin_ft >= ASR_base:
        print(
            "⚠️ La mejor poda encontrada no reduce ASR (o incluso lo empeora). "
            "Aun así se aplica fine-pruning por consistencia, "
            "pero podrías optar por no aplicar defensa en este caso."
        )

    print(
        f"Usando prune_ratio_opt={prune_ratio_opt:.2f} "
        f"(pruned {num_pruned_opt}/{num_channels} canales)"
    )

    # 4) Construir modelo podado final + fine-tuning limpio
    fp_model = load_backdoor_model(model_path)
    fp_model, gate = attach_gate_after_layer4(fp_model, mask_opt)
    fp_model.to(DEVICE)

    print("Fine-tuning del modelo podado sobre datos limpios...")
    fine_tune(fp_model, ft_loader)

    # 5) Evaluar modelo defendido
    ATA_after = evaluate_accuracy(fp_model, test_loader)
    ASR_after = evaluate_asr(
        fp_model, test_loader, target, trigger_components
    )

    print(
        f"[Fine-Pruned] ATA_def (clean test): {ATA_after * 100:.2f}%, "
        f"ASR_def: {ASR_after * 100:.2f}%"
    )

    # 6) Guardar modelo defendido
    model_base = os.path.basename(model_path)
    root, ext = os.path.splitext(model_base)
    out_model_name = f"{root}_finepruned_{int(prune_ratio_opt * 100)}{ext}"
    out_model_path = os.path.join(OUT_DIR, out_model_name)
    torch.save(fp_model.state_dict(), out_model_path)
    print(f"✅ Modelo defendido guardado en: {out_model_path}")

    # 7) Devolver resultados para log
    return {
        "file": base_name + ".h5",
        "model": model_base,
        "def_model": out_model_name,
        "poison_rate": info["poison_rate"],
        "target": target,
        "blur_type": info["blur_type"],
        "threshold": info["threshold"],
        "csv_poison_count": csv_poison_count,
        "num_train": num_train,
        "prune_ratio_opt": prune_ratio_opt,
        "num_channels": num_channels,
        "num_pruned": num_pruned_opt,
        "ATA": ATA_base,
        "ASR": ASR_base,
        "ATA_podada_sin_ft": ATA_opt_sin_ft,
        "ASR_podada_sin_ft": ASR_opt_sin_ft,
        "ATA_def": ATA_after,
        "ASR_def": ASR_after,
    }


# ============================================================
# 7. Main: recorrer todos los H5 poison + modelos
# ============================================================

if __name__ == "__main__":
    # 1) Datos limpios (para activaciones, fine-tuning y test)
    print("Cargando datos limpios desde:", CLEAN_H5)
    val_loader, ft_loader, test_loader = load_clean_data(CLEAN_H5)

    # 2) Listar todos los H5 poison
    pattern_h5 = os.path.join(POISON_H5_DIR, "gtsrb-sub18-pois-*-t*-*.h5")
    h5_paths = sorted(glob.glob(pattern_h5))

    if not h5_paths:
        print("⚠️ No se encontraron H5 envenenados con el patrón:", pattern_h5)
        raise SystemExit

    print(f"Encontrados {len(h5_paths)} H5 envenenados.")

    results = []

    for h5_path in h5_paths:
        info = parse_poison_filename(h5_path)
        base_name = info["base_name"]

        # Construir el nombre del modelo envenenado
        model_name = MODEL_PREFIX + base_name + MODEL_SUFFIX
        model_path = os.path.join(MODELS_DIR, model_name)

        if not os.path.exists(model_path):
            print(f"⚠️ Modelo no encontrado para {base_name}: {model_path}")
            continue

        try:
            res = run_fine_pruning_with_asr_for_pair(
                model_path, h5_path, val_loader, ft_loader, test_loader
            )
            results.append(res)
        except Exception as e:
            print(f"❌ Error procesando {base_name}: {e}")

    # 3) Guardar CSV resumen
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\n📄 Resultados guardados en: {RESULTS_CSV}")
    else:
        print("⚠️ No se generaron resultados (¿no se encontró ningún modelo?).")
