# ============================================================
# 🚀 ResNet18 (PyTorch) - Evaluación de datasets envenenados (.h5)
# VERSIÓN 64x64 / SÓLO TRAIN+TEST / EARLY STOP EN TRAIN LOSS
# + DEFENSA PiDAn (por clase target) + MÉTRICAS TP/FP/FN vs CSV
# + BARRIDO CPV_THRESH ∈ {0.94, 0.95, 0.96, 0.97}
# 🔧 ADAPTADO PARA USAR MODELOS BACKDOOR PRE-ENTRENADOS (NO REENTRENA EL ATAQUE)
# ============================================================

import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from collections import defaultdict
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
import pandas as pd  # para leer _poisoned_rows.csv

# ------------------------------------------------------------
# ⚙️ Configuración general
# ------------------------------------------------------------
ROOT = "GTSRBzip/poisoned_from_torchvision_64x64_train_test"

# *** CAMBIO: carpeta donde ya tienes los modelos backdoor entrenados ***
BACKDOOR_MODELS_DIR = "Modelosenvenenadosgotas"

EPOCHS = 35
BATCH_SIZE = 128
LR = 1e-3

# 🔁 CPV a barrer
CPV_THRESH_VALUES = [0.94, 0.95, 0.96, 0.97]

RESULTS_JSON = "resultados_envenenamiento_ResNet18_64x64_train_only_PiDAn_CPV_lluvia.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ------------------------------------------------------------
# 💧💧 Funciones del Trigger (igual que generador) 💧💧
# ------------------------------------------------------------
def create_rain_trigger_advanced(shape, k1_size=5, k2_size=21, mask_threshold=150,
                                 refraction_intensity=0.04,
                                 use_motion_blur=False,
                                 motion_blur_strength=15,
                                 motion_blur_angle=45):
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


def apply_trigger_with_advanced_effects(image_original_bgr, trigger_components,
                                        specular_intensity=25, blend_alpha=0.25):
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


# ------------------------------------------------------------
# 🔧 EarlyStopping basado en *train_loss* (para modelos defendidos)
# ------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=6, delta=1e-4, save_path="best_temp.pth"):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_state = None
        self.save_path = save_path

    def __call__(self, train_loss, model):
        if train_loss < self.best_loss - self.delta:
            self.best_loss = train_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ------------------------------------------------------------
# 💧 Cálculo del Attack Success Rate (ASR)
# ------------------------------------------------------------
def compute_asr(model, test_poisoned_loader, remapped_target_class, device):
    """
    ASR = % de imágenes de test que NO eran target y que son clasificadas como target.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels_clean in test_poisoned_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            is_target = (preds == remapped_target_class)
            was_not_target = (labels_clean.to(device) != remapped_target_class)

            correct += (is_target & was_not_target).sum().item()
            total += was_not_target.sum().item()

    return correct / total if total > 0 else 0.0


# ------------------------------------------------------------
# 🧬 Helpers para extraer features de ResNet18 y aplicar PiDAn
# ------------------------------------------------------------
def extract_features_resnet18(model, x):
    """
    Extrae el penúltimo layer de resnet18 (vector de dimensión 512).
    x: tensor (B, 3, H, W) ya normalizado como en el entrenamiento.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)  # (B, 512)
    return x


def extract_features_dataset(model, dataloader, device):
    """
    Recorre un DataLoader y devuelve:
      - feats: np.array (N, D)
      - labels: np.array (N,)
    """
    model.eval()
    feats_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="PiDAn - extrayendo features"):
            imgs = imgs.to(device)
            f = extract_features_resnet18(model, imgs)
            feats_list.append(f.cpu().numpy())
            labels_list.append(labels.numpy())

    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels


def l2_normalize_rows(X, eps=1e-8):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms


def compute_P_subspace(X_c, cpv_thresh=0.95):
    """
    X_c: (m_c, D)  features centralizadas y normalizadas de una clase.
    """
    m_c, D = X_c.shape
    Sigma = (X_c.T @ X_c) / m_c  # (D, D)

    eigvals, eigvecs = np.linalg.eigh(Sigma)  # ascendentes
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    k = np.searchsorted(cumvar, cpv_thresh) + 1
    P = eigvecs[:, :k]   # (D, k)
    return P, eigvals[:k], k


def pidan_weights_for_class(X_c, P):
    """
    X_c: (m_c, D)
    P: (D, k)
    Devuelve a*: (m_c,) vector de pesos PiDAn para esa clase.
    """
    m_c, D = X_c.shape
    X = X_c.T                      # (D, m_c)
    I = np.eye(D)
    proj = I - P @ P.T             # (D, D)
    M = X.T @ (proj @ X)           # (m_c, m_c)

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argmax(eigvals)
    a_star = eigvecs[:, idx]
    return a_star


def pidan_split_clean_poison_for_class(a_star, mask_class_bool):
    """
    a_star: (m_c,) pesos PiDAn de la clase c.
    mask_class_bool: máscara booleana de longitud N (train completo)
    """
    a_vals = a_star.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = kmeans.fit_predict(a_vals)   # 0 o 1

    counts = np.bincount(labels)
    poison_cluster = np.argmin(counts)    # cluster pequeño -> poison

    mask_local_poison = (labels == poison_cluster)
    mask_local_clean = ~mask_local_poison

    idx_global = np.where(mask_class_bool)[0]
    idx_poison_global = idx_global[mask_local_poison]
    idx_clean_global = idx_global[mask_local_clean]

    return idx_clean_global, idx_poison_global


# ------------------------------------------------------------
# 🔍 Detectar y PARSEAR datasets envenenados (.h5)
# ------------------------------------------------------------
files = [f for f in os.listdir(ROOT) if f.endswith(".h5") and "pois" in f]
pattern = re.compile(r"pois-(\d+)-t(\d+)-(mb(\d+)a(\d+)|k2(\d+))-th(\d+)\.h5")

parsed_files = []
for f in files:
    match = pattern.search(f)
    if match:
        data = {
            "file_name": f,
            "poison_rate": int(match.group(1)),   # en %
            "target": int(match.group(2)),
            "blur_full": match.group(3),
            "threshold": int(match.group(7)),
        }

        if data["blur_full"].startswith("mb"):
            data["use_motion_blur"] = True
            data["mb_strength"] = int(match.group(4))
            data["mb_angle"] = int(match.group(5))
            data["k2_size"] = 21
        else:
            data["use_motion_blur"] = False
            data["k2_size"] = int(match.group(6))
            data["mb_strength"] = 15
            data["mb_angle"] = 45

        parsed_files.append(data)
    else:
        print(f"⚠️ Aviso: Se omitirá el archivo (nombre no estándar): {f}")

parsed_files.sort(key=lambda x: (x["blur_full"], x["threshold"], x["poison_rate"]))

if not parsed_files:
    raise FileNotFoundError(f"❌ No se encontraron datasets .h5 válidos en: {ROOT}")

print(f"\n📂 Se detectaron y parsearon {len(parsed_files)} datasets envenenados.")
results = []


# ------------------------------------------------------------
# 🚀 Loop principal por dataset
# ------------------------------------------------------------
for file_data in parsed_files:

    file_name = file_data["file_name"]
    target_class_original = file_data["target"]

    path = os.path.join(ROOT, file_name)
    print(f"\n==============================")
    print(f"📁 Procesando: {file_name}")

    # ---------------- Cargar datos (SOLO train y test) ----------------
    with h5py.File(path, "r") as f:
        x_train = np.array(f["x_train"])  # (N_tr, 64,64,3) en [0,1]
        y_train = np.array(f["y_train"])
        x_test = np.array(f["x_test"])
        y_test = np.array(f["y_test"])

    IMG_HEIGHT, IMG_WIDTH = x_test.shape[1:3]

    # --- Mapeo de clases (por seguridad) ---
    unique_classes = np.unique(y_train)
    class_map = {orig: idx for idx, orig in enumerate(unique_classes)}
    y_train_mapped = np.array([class_map[y] for y in y_train])
    y_test_mapped = np.array([class_map[y] for y in y_test])
    num_classes = len(unique_classes)

    if target_class_original not in class_map:
        print(f"⚠️ Aviso: La clase objetivo {target_class_original} no está en el set. Saltando.")
        continue
    remapped_target = class_map[target_class_original]
    print(f"Clases: {num_classes}. Target {target_class_original} -> remapeado a {remapped_target}")

    # ---------------- Tensores y DataLoaders ----------------
    def to_tensor_x(arr):
        t = torch.tensor(arr.transpose(0, 3, 1, 2), dtype=torch.float32)
        t = (t - 0.5) * 2.0  # Normalize(mean=0.5, std=0.5) equivalente
        return t

    x_train_tensor = to_tensor_x(x_train)
    x_test_clean_tensor = to_tensor_x(x_test)

    y_train_tensor = torch.tensor(y_train_mapped, dtype=torch.long)
    y_test_clean_tensor = torch.tensor(y_test_mapped, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_clean_loader = DataLoader(
        TensorDataset(x_test_clean_tensor, y_test_clean_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --- Generar Test Envenenado (x_test_poisoned) ---
    print("💧 Envenenando el Test-Set en memoria para calcular ASR...")

    trigger_components = create_rain_trigger_advanced(
        (IMG_HEIGHT, IMG_WIDTH),
        k1_size=5,
        k2_size=file_data["k2_size"],
        mask_threshold=file_data["threshold"],
        refraction_intensity=0.04,
        use_motion_blur=file_data["use_motion_blur"],
        motion_blur_strength=file_data["mb_strength"],
        motion_blur_angle=file_data["mb_angle"],
    )

    x_test_poisoned = np.zeros_like(x_test)
    for i in tqdm(range(len(x_test)), desc="Aplicando trigger a x_test"):
        x_orig_rgb = (x_test[i] * 255).astype(np.uint8)
        x_orig_bgr = cv2.cvtColor(x_orig_rgb, cv2.COLOR_RGB2BGR)

        poisoned_bgr = apply_trigger_with_advanced_effects(
            x_orig_bgr,
            trigger_components,
            specular_intensity=25,
            blend_alpha=0.5,
        )

        poisoned_rgb = cv2.cvtColor(poisoned_bgr, cv2.COLOR_BGR2RGB)
        x_test_poisoned[i] = poisoned_rgb.astype(np.float32) / 255.0

    x_test_poisoned_tensor = to_tensor_x(x_test_poisoned)
    test_poisoned_loader = DataLoader(
        TensorDataset(x_test_poisoned_tensor, y_test_clean_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    print("✅ Test-Set envenenado listo.")

    # ---------------- Modelo BACKDOOR (CARGADO, NO ENTRENADO AQUÍ) ----------------
    backdoor_model_path = os.path.join(
        BACKDOOR_MODELS_DIR,
        f"best_model_{file_name}.pth"  # mismo patrón que usaste al guardar
    )

    if not os.path.exists(backdoor_model_path):
        print(f"❌ No se encontró modelo backdoor para {file_name}: {backdoor_model_path}")
        print("   Se salta este dataset.")
        continue

    print(f"📥 Cargando modelo backdoor desde: {backdoor_model_path}")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    sd = torch.load(backdoor_model_path, map_location=device)

    # por si en algún momento guardaste un dict con 'state_dict' o 'model_state_dict'
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    model.load_state_dict(sd)
    model = model.to(device)

    # ---------------- Evaluación en test limpio (ATA) ----------------
    model.eval()
    y_true_clean, y_pred_clean = [], []
    with torch.no_grad():
        for imgs, labels in test_clean_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true_clean.extend(labels.cpu().numpy())
            y_pred_clean.extend(preds.cpu().numpy())
    ATA = (np.array(y_pred_clean) == np.array(y_true_clean)).mean()

    # ---------------- Evaluación de ASR (modelo backdoor) ----------------
    ASR = compute_asr(model, test_poisoned_loader, remapped_target, device)

    print(f"✅ [BACKDOOR pre-entrenado] ATA (Test limpio): {ATA * 100:.2f}% "
          f"| 🎯 ASR (Ataque): {ASR * 100:.2f}%")

    # =========================
    # 🛡️ Defensa PiDAn en este dataset (barrido CPV_THRESH)
    # =========================
    print("🛡️ Aplicando PiDAn (barrido CPV) para detectar y filtrar ejemplos envenenados...")

    # 1) DataLoader sin shuffle para extraer features del train
    train_loader_eval = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2) Extraer features del penúltimo layer de ResNet18 (UNA vez)
    feats_train, labels_train_feat = extract_features_dataset(
        model, train_loader_eval, device
    )
    feats_test, labels_test_feat = extract_features_dataset(
        model, test_clean_loader, device
    )

    # 3) Centralizar con la media del test limpio y normalizar filas
    mu_clean = feats_test.mean(axis=0, keepdims=True)
    X_train = l2_normalize_rows(feats_train - mu_clean)
    X_test = l2_normalize_rows(feats_test - mu_clean)

    N_train = len(y_train_mapped)

    # Máscara de la clase target (remapeada)
    mask_c = (labels_train_feat == remapped_target)
    X_c_base = X_train[mask_c]

    # Cargar CSV con ground truth de índices envenenados
    csv_poison_count = None
    gt_set = None
    csv_path = os.path.join(ROOT, file_name.replace(".h5", "_poisoned_rows.csv"))
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        gt_indices = df_csv["index"].to_numpy(dtype=int)
        gt_set = set(gt_indices.tolist())
        csv_poison_count = len(gt_set)
        print(f"📑 CSV poison total: {csv_poison_count}")
    else:
        print(f"⚠️ No se encontró CSV de veneno para este dataset: {csv_path}")

    # Si hay muy pocos ejemplos de la clase target, no aplicamos PiDAn
    if len(X_c_base) < 5:
        print("⚠️ Muy pocos ejemplos de la clase target para aplicar PiDAn. Se omite defensa.")
        for cpv in CPV_THRESH_VALUES:
            results.append({
                "file": file_name,
                "poison_rate": file_data["poison_rate"],
                "target": target_class_original,
                "blur_type": file_data["blur_full"],
                "threshold": file_data["threshold"],
                "CPV_THRESH": float(cpv),
                "ATA": float(ATA),
                "ASR": float(ASR),
                "ATA_def": float(ATA),
                "ASR_def": float(ASR),
                "num_removed": 0,
                "num_train": int(N_train),
                "csv_poison_count": int(csv_poison_count) if csv_poison_count is not None else None,
                "pidan_TP": 0,
                "pidan_FP": 0,
                "pidan_FN": 0,
                "pidan_precision": None,
                "pidan_recall": None,
            })
        continue

    # =========================
    # 🔁 BARRIDO SOBRE CPV_THRESH_VALUES
    # =========================
    for cpv in CPV_THRESH_VALUES:
        print(f"\n🧪 PiDAn (KMeans) con CPV_THRESH = {cpv:.2f}")

        # Subespacio P con este CPV
        X_c = X_c_base
        P, _, _ = compute_P_subspace(X_c, cpv_thresh=cpv)
        a_star = pidan_weights_for_class(X_c, P)

        # Separar clean/poison en la clase target (KMeans)
        idx_clean_c, idx_poison_c = pidan_split_clean_poison_for_class(a_star, mask_c)

        # Máscara global de qué muestras de train conservamos
        mask_keep = np.ones_like(labels_train_feat, dtype=bool)
        mask_keep[idx_poison_c] = False

        num_removed = (~mask_keep).sum()
        print(
            f"PiDAn (CPV={cpv:.2f}) marcó {num_removed} muestras como poison "
            f"({num_removed / len(mask_keep):.2%} del train)."
        )

        # --------- Métricas PiDAn vs CSV (si tenemos GT) ---------
        pidan_tp = pidan_fp = pidan_fn = 0
        pidan_prec = pidan_rec = None

        if gt_set is not None:
            pred_set = set(idx_poison_c.tolist())

            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            pidan_tp, pidan_fp, pidan_fn = tp, fp, fn

            if tp + fp > 0:
                pidan_prec = tp / (tp + fp)
            if tp + fn > 0:
                pidan_rec = tp / (tp + fn)

            print(f"PiDAn vs CSV -> TP: {tp}, FP: {fp}, FN: {fn}")
            if pidan_prec is not None and pidan_rec is not None:
                print(f"Precision: {pidan_prec*100:.2f}%, Recall: {pidan_rec*100:.2f}%")
        else:
            print("Sin CSV -> no se calculan TP/FP/FN.")

        # --------- Re-entrenar modelo DEFENDIDO para este CPV ---------
        x_train_def_tensor = x_train_tensor[mask_keep]
        y_train_def = y_train_mapped[mask_keep]
        y_train_def_tensor = torch.tensor(y_train_def, dtype=torch.long)

        train_loader_def = DataLoader(
            TensorDataset(x_train_def_tensor, y_train_def_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        model_def = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_def.fc = nn.Linear(model_def.fc.in_features, num_classes)
        model_def = model_def.to(device)

        criterion_def = nn.CrossEntropyLoss()
        optimizer_def = optim.Adam(model_def.parameters(), lr=LR)
        model_def_save_path = f"best_model_DEF_{file_name}_cpv{int(cpv*100)}.pth"
        early_stopping_def = EarlyStopping(patience=6, save_path=model_def_save_path)

        print(f"Iniciando entrenamiento DEFENDIDO (PiDAn, CPV={cpv:.2f})...")
        for epoch in range(EPOCHS):
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
            print(
                f"[DEF CPV={cpv:.2f}] E {epoch + 1:02d} | Train Acc: {epoch_acc_def * 100:.2f}% "
                f"| Train Loss: {epoch_loss_def:.4f}"
            )

            early_stopping_def(epoch_loss_def, model_def)
            if early_stopping_def.early_stop:
                print(f"🛑 [DEF CPV={cpv:.2f}] Early stopping en epoch {epoch + 1} (train_loss)")
                break

        print(f"Cargando mejor modelo DEFENDIDO (CPV={cpv:.2f}) desde: {model_def_save_path}")
        model_def.load_state_dict(torch.load(model_def_save_path, map_location=device))
        model_def = model_def.to(device)
        model_def.eval()

        # 6) Evaluar ATA_def (test limpio)
        y_true_clean_def, y_pred_clean_def = [], []
        with torch.no_grad():
            for imgs, labels in test_clean_loader:
                imgs = imgs.to(device)
                outputs = model_def(imgs)
                _, preds = torch.max(outputs, 1)
                y_true_clean_def.extend(labels.cpu().numpy())
                y_pred_clean_def.extend(preds.cpu().numpy())
        ATA_def = (np.array(y_pred_clean_def) == np.array(y_true_clean_def)).mean()

        # y ASR_def usando el mismo test_poisoned_loader
        ASR_def = compute_asr(model_def, test_poisoned_loader, remapped_target, device)

        print(
            f"🛡️ DEF PiDAn (CPV={cpv:.2f}) -> ATA_def: {ATA_def * 100:.2f}% "
            f"| ASR_def: {ASR_def * 100:.2f}%"
        )

        # Guardar resultados de este dataset + CPV
        results.append({
            "file": file_name,
            "poison_rate": file_data["poison_rate"],
            "target": target_class_original,
            "blur_type": file_data["blur_full"],
            "threshold": file_data["threshold"],
            "CPV_THRESH": float(cpv),
            "ATA": float(ATA),
            "ASR": float(ASR),
            "ATA_def": float(ATA_def),
            "ASR_def": float(ASR_def),
            "num_removed": int(num_removed),
            "num_train": int(N_train),
            "csv_poison_count": int(csv_poison_count) if csv_poison_count is not None else None,
            "pidan_TP": int(pidan_tp),
            "pidan_FP": int(pidan_fp),
            "pidan_FN": int(pidan_fn),
            "pidan_precision": float(pidan_prec) if pidan_prec is not None else None,
            "pidan_recall": float(pidan_rec) if pidan_rec is not None else None,
        })

# ------------------------------------------------------------
# 💾 Guardar resumen JSON
# ------------------------------------------------------------
json.dump(results, open(RESULTS_JSON, "w"), indent=2)
print(f"\n💾 Resultados completos guardados en {RESULTS_JSON}")

# ------------------------------------------------------------
# 📊 Gráficos ATA vs ASR para cada configuración de trigger + CPV
# ------------------------------------------------------------
print("Generando gráficos de resultados...")
grouped_results = defaultdict(list)
for r in results:
    key = (r["blur_type"], r["threshold"], r["CPV_THRESH"])
    grouped_results[key].append(r)

for (blur, threshold, cpv), group_data in grouped_results.items():
    group_data.sort(key=lambda x: x["poison_rate"])
    rates = [r["poison_rate"] for r in group_data]
    ata = [r["ATA"] * 100 for r in group_data]
    asr = [r["ASR"] * 100 for r in group_data]
    ata_def = [r["ATA_def"] * 100 for r in group_data]
    asr_def = [r["ASR_def"] * 100 for r in group_data]

    plt.figure(figsize=(8, 6))
    plt.plot(rates, ata, "o-", color="green", label="ATA (test limpio)")
    plt.plot(rates, asr, "o--", color="red", label="ASR (ataque)")
    plt.plot(rates, ata_def, "s-", color="blue", label=f"ATA defendido PiDAn (CPV={cpv:.2f})")
    plt.plot(rates, asr_def, "s--", color="orange", label=f"ASR defendido PiDAn (CPV={cpv:.2f})")
    plt.xlabel("Porcentaje de envenenamiento (%)")
    plt.ylabel("Precisión (%)")
    plot_title = (
        f"Efecto del veneno + PiDAn - ResNet18\n"
        f"Trigger: {blur}, Threshold: {threshold}, CPV={cpv:.2f}"
    )
    plt.title(plot_title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"ATA_ASR_ResNet18_64x64_train_only_PiDAn_cpv{int(cpv*100)}_{blur}_th{threshold}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"📊 Gráfico guardado: {plot_filename}")
    plt.close()

print("\n--- Proceso completado ---")
