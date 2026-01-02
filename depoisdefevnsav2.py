# ============================================================
# 🛡️ De-Pois-lite para GTSRB 18 clases (unknown-κ)
# - Teacher: ResNet18 entrenado con cada dataset envenenado
# - Mimic: ResNet18 limpio cargado desde .pth
# - Selección de poison con GMM 1D sobre scores (sin usar poison_rate)
# 🔧 ADAPTADO PARA USAR MODELOS BACKDOOR PRE-ENTRENADOS (NO REENTRENA EL ATAQUE)
# ============================================================

import os
import re
import json

import h5py
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# ⚙️ Configuración general
# ------------------------------------------------------------
ROOT_POISON = "GTSRBzip/poisoned_from_torchvision_64x64_train_test"
CLEAN_MODEL_PATH = "GTSRBzip/resnet18_gtsrb_sub18_clean.pth"  # 👈 tu modelo limpio

NUM_CLASSES_SUB18 = 18           # número de clases del sub18

# *** CAMBIO: carpeta donde ya tienes los modelos backdoor entrenados ***
BACKDOOR_MODELS_DIR = "Modelosenvenenadosgotas"

EPOCHS_DEF = 30
BATCH_SIZE = 128
LR_DEF = 1e-3

RESULTS_JSON = "resultados_DePois_mimic_cleanModel_lluvia.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)


# ------------------------------------------------------------
# 🔧 Normalización de imágenes
# ------------------------------------------------------------
def to_tensor_x(arr):
    """
    arr: (N, H, W, C) en [0,1]
    salida: tensor (N, C, H, W) en [-1,1]
    """
    t = torch.tensor(arr.transpose(0, 3, 1, 2), dtype=torch.float32)
    t = (t - 0.5) * 2.0
    return t


# ------------------------------------------------------------
# 💧💧 Trigger de lluvia (como en tus scripts) 💧💧
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
                                        specular_intensity=25, blend_alpha=0.5):
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
# 🔧 EarlyStopping (para modelo DEFENDIDO)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 🧠 Modelos
# ------------------------------------------------------------
def build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_clean_model(num_classes):
    """
    Carga el modelo limpio desde CLEAN_MODEL_PATH.
    Asumimos que el .pth es un state_dict compatible con build_resnet18(num_classes).
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
# 🧪 Entrenamiento DEFENDIDO (se mantiene igual)
# ------------------------------------------------------------
def train_defended_model(x_train_def_tensor, y_train_def, num_classes, save_path):
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

    print("Iniciando entrenamiento DEFENDIDO (De-Pois-lite)...")
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


# ------------------------------------------------------------
# 📏 Scoring De-Pois-lite (teacher vs modelo limpio)
# ------------------------------------------------------------
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
        for imgs, labels in tqdm(loader, desc="De-Pois scoring"):
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


# ------------------------------------------------------------
# 🧮 Selección unknown-κ con GMM 1D
# ------------------------------------------------------------
def select_poison_gmm(scores,
                      min_fraction=0.01,
                      max_fraction=0.4,
                      prob_thresh=0.8,
                      random_state=0):
    """
    scores: np.array shape (N,)
    Devuelve:
      - idx_poison: índices predichos como poison
      - probs_poison: probabilidad de pertenecer al componente "poison"
      - frac_est: fracción estimada de poison
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

    # máscara inicial por probabilidad
    mask_poison = probs > prob_thresh
    n_poison = mask_poison.sum()

    min_count = max(1, int(round(min_fraction * N)))
    max_count = int(round(max_fraction * N))

    if n_poison < min_count:
        # si hay muy pocas, cogemos las top min_count por probabilidad
        top_idx = np.argsort(probs)[::-1][:min_count]
        mask_poison[:] = False
        mask_poison[top_idx] = True
        n_poison = min_count
    elif n_poison > max_count:
        # si hay demasiadas, recortamos a las top max_count
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


# ------------------------------------------------------------
# 🔍 Detectar y parsear datasets envenenados (.h5)
# ------------------------------------------------------------
files = [f for f in os.listdir(ROOT_POISON) if f.endswith(".h5") and "pois" in f]
pattern = re.compile(r"pois-(\d+)-t(\d+)-(mb(\d+)a(\d+)|k2(\d+))-th(\d+)\.h5")

parsed_files = []
for f in files:
    m = pattern.search(f)
    if m:
        data = {
            "file_name": f,
            "poison_rate": int(m.group(1)),  # solo para logging / análisis
            "target": int(m.group(2)),
            "blur_full": m.group(3),
            "mb_strength": int(m.group(4)) if m.group(4) is not None else 15,
            "mb_angle": int(m.group(5)) if m.group(5) is not None else 45,
            "k2_size": int(m.group(6)) if m.group(6) is not None else 21,
            "threshold": int(m.group(7)),
        }
        data["use_motion_blur"] = data["blur_full"].startswith("mb")
        parsed_files.append(data)
    else:
        print("⚠️ Archivo omitido (nombre no estándar):", f)

parsed_files.sort(key=lambda x: (x["blur_full"], x["threshold"], x["poison_rate"]))

if not parsed_files:
    raise FileNotFoundError(f"No se encontraron datasets .h5 válidos en {ROOT_POISON}")

print(f"\n📂 Se detectaron {len(parsed_files)} datasets envenenados.")

# Cargamos el modelo limpio UNA VEZ
clean_model = load_clean_model(NUM_CLASSES_SUB18)

results = []


# ------------------------------------------------------------
# 🚀 Loop principal por dataset envenenado
# ------------------------------------------------------------
for file_data in parsed_files:
    file_name = file_data["file_name"]
    poison_rate_real = file_data["poison_rate"]  # sólo informativo
    target_class = file_data["target"]

    path = os.path.join(ROOT_POISON, file_name)
    print("\n==============================")
    print("📁 Procesando:", file_name)

    # 1) Cargar datos envenenados
    with h5py.File(path, "r") as f:
        x_train_poison = np.array(f["x_train"])   # (N, 64,64,3) en [0,1]
        y_train_poison = np.array(f["y_train"])
        x_test = np.array(f["x_test"])
        y_test = np.array(f["y_test"])

    x_train_poison_tensor = to_tensor_x(x_train_poison)
    x_test_clean_tensor = to_tensor_x(x_test)

    y_train_poison_tensor = torch.tensor(y_train_poison, dtype=torch.long)
    y_test_clean_tensor = torch.tensor(y_test, dtype=torch.long)

    unique_classes = np.unique(y_train_poison)
    print("Clases en el train envenenado:", unique_classes)
    if len(unique_classes) != NUM_CLASSES_SUB18:
        print("⚠️ Aviso: número de clases distinto a NUM_CLASSES_SUB18 (ajusta si hace falta).")

    # 2) DataLoader test limpio
    test_clean_loader = DataLoader(
        TensorDataset(x_test_clean_tensor, y_test_clean_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 3) Generar TEST envenenado con trigger de lluvia
    IMG_HEIGHT, IMG_WIDTH = x_test.shape[1:3]
    trig = create_rain_trigger_advanced(
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
        x_rgb = (x_test[i] * 255).astype(np.uint8)
        x_bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

        poisoned_bgr = apply_trigger_with_advanced_effects(
            x_bgr, trig, specular_intensity=25, blend_alpha=0.5
        )

        poisoned_rgb = cv2.cvtColor(poisoned_bgr, cv2.COLOR_BGR2RGB)
        x_test_poisoned[i] = poisoned_rgb.astype(np.float32) / 255.0

    x_test_poisoned_tensor = to_tensor_x(x_test_poisoned)
    test_poisoned_loader = DataLoader(
        TensorDataset(x_test_poisoned_tensor, y_test_clean_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 4) Cargar modelo BACKDOOR ya entrenado (NO reentrenar)
    backdoor_model_path = os.path.join(
        BACKDOOR_MODELS_DIR,
        f"best_model_{file_name}.pth"   # mismo patrón que en tus otros scripts
    )
    if not os.path.exists(backdoor_model_path):
        print(f"❌ No se encontró modelo backdoor para {file_name}: {backdoor_model_path}")
        print("   Se salta este dataset.")
        continue

    print(f"📥 Cargando modelo BACKDOOR desde: {backdoor_model_path}")
    backdoor_model = build_resnet18(NUM_CLASSES_SUB18).to(device)
    sd = torch.load(backdoor_model_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    backdoor_model.load_state_dict(sd)
    backdoor_model = backdoor_model.to(device)

    # 5) Evaluar ATA y ASR del modelo backdoor
    backdoor_model.eval()
    y_true_clean, y_pred_clean = [], []
    with torch.no_grad():
        for imgs, labels in test_clean_loader:
            imgs = imgs.to(device)
            outputs = backdoor_model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true_clean.extend(labels.numpy())
            y_pred_clean.extend(preds.cpu().numpy())
    ATA = (np.array(y_true_clean) == np.array(y_pred_clean)).mean()

    remapped_target = target_class  # asumimos que ya está en el mismo espacio 0..17
    ASR = compute_asr(backdoor_model, test_poisoned_loader, remapped_target, device)

    print(f"✅ BACKDOOR (pre-entrenado) -> ATA: {ATA*100:.2f}% | ASR: {ASR*100:.2f}% "
          f"(poison_rate real = {poison_rate_real}%)")

    # 6) Scoring De-Pois-lite usando modelo LIMPIO como mimic
    scores = score_poison_with_clean_model(
        x_train_poison_tensor, y_train_poison,
        backdoor_model, clean_model
    )

    # 7) Selección unknown-κ con GMM
    idx_poison_pred, probs_poison, frac_est = select_poison_gmm(
        scores,
        min_fraction=0.01,
        max_fraction=0.5,
        prob_thresh=0.9,
        random_state=0
    )

    N = len(scores)
    mask_keep = np.ones(N, dtype=bool)
    mask_keep[idx_poison_pred] = False
    num_removed = (~mask_keep).sum()
    print(f"De-Pois-lite GMM marcó {num_removed} muestras como poison "
          f"({num_removed/N:.2%} del train, κ_est ≈ {frac_est*100:.2f}%).")

    # 8) Métricas vs CSV (_poisoned_rows.csv)
    csv_poison_count = None
    dep_tp = dep_fp = dep_fn = None
    dep_prec = dep_rec = None

    csv_path = os.path.join(ROOT_POISON, file_name.replace(".h5", "_poisoned_rows.csv"))
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        gt_indices = df["index"].to_numpy(dtype=int)
        gt_set = set(gt_indices.tolist())
        pred_set = set(idx_poison_pred.tolist())
        csv_poison_count = len(gt_set)

        dep_tp = len(gt_set & pred_set)
        dep_fp = len(pred_set - gt_set)
        dep_fn = len(gt_set - pred_set)

        if dep_tp + dep_fp > 0:
            dep_prec = dep_tp / (dep_tp + dep_fp)
        if dep_tp + dep_fn > 0:
            dep_rec = dep_tp / (dep_tp + dep_fn)

        print(f"📑 CSV poison total: {csv_poison_count}")
        print(f"De-Pois-lite (GMM) vs CSV -> TP: {dep_tp}, FP: {dep_fp}, FN: {dep_fn}")
        if dep_prec is not None and dep_rec is not None:
            print(f"Precision: {dep_prec*100:.2f}% | Recall: {dep_rec*100:.2f}%")
    else:
        print("⚠️ No se encontró CSV de veneno para este dataset, se omiten TP/FP/FN.")

    # 9) Re-entrenar modelo DEFENDIDO con las muestras 'keep'
    x_train_def_tensor = x_train_poison_tensor[mask_keep]
    y_train_def = y_train_poison[mask_keep]

    def_ckpt = f"best_defended_{file_name}_depois_gmm.pth"
    defended_model = train_defended_model(
        x_train_def_tensor, y_train_def,
        NUM_CLASSES_SUB18, def_ckpt
    )

    # 10) Evaluar ATA_def y ASR_def
    defended_model.eval()
    y_true_clean_def, y_pred_clean_def = [], []
    with torch.no_grad():
        for imgs, labels in test_clean_loader:
            imgs = imgs.to(device)
            outputs = defended_model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true_clean_def.extend(labels.numpy())
            y_pred_clean_def.extend(preds.cpu().numpy())
    ATA_def = (np.array(y_true_clean_def) == np.array(y_pred_clean_def)).mean()

    ASR_def = compute_asr(defended_model, test_poisoned_loader, remapped_target, device)

    print(f"🛡️ DEF De-Pois-lite (GMM) -> ATA_def: {ATA_def*100:.2f}% | ASR_def: {ASR_def*100:.2f}%")

    # 11) Registrar resultados
    results.append({
        "file": file_name,
        "poison_rate_real": poison_rate_real,
        "target": target_class,
        "blur_type": file_data["blur_full"],
        "threshold": file_data["threshold"],
        "ATA": float(ATA),
        "ASR": float(ASR),
        "ATA_def": float(ATA_def),
        "ASR_def": float(ASR_def),
        "num_removed": int(num_removed),
        "num_train": int(N),
        "kappa_estimated": float(frac_est),
        "csv_poison_count": int(csv_poison_count) if csv_poison_count is not None else None,
        "depois_TP": int(dep_tp) if dep_tp is not None else None,
        "depois_FP": int(dep_fp) if dep_fp is not None else None,
        "depois_FN": int(dep_fn) if dep_fn is not None else None,
        "depois_precision": float(dep_prec) if dep_prec is not None else None,
        "depois_recall": float(dep_rec) if dep_rec is not None else None,
    })


# ------------------------------------------------------------
# 💾 Guardar resultados en JSON
# ------------------------------------------------------------
json.dump(results, open(RESULTS_JSON, "w"), indent=2)
print("\n💾 Resultados completos guardados en", RESULTS_JSON)
print("\n--- Proceso De-Pois-lite (unknown-κ con GMM) completado ---")
