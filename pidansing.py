# ============================================================
# 🚀 Ataque SIG + Defensa PiDAn en GTSRB 64x64 (frecuencia f=5)
#   - Usa mismos hiperparámetros y trigger que tu script de ataque
#   - CARGA modelo con backdoor SIG ya entrenado (NO reentrena)
#   - Calcula ACC limpia y ASR_SIG
#   - Aplica PiDAn sobre la clase target (SIG_TARGET_LABEL)
#   - Reentrena modelo defendido y mide ACC / ASR_SIG defendidos
#   - Barrido CPV_THRESH ∈ {0.94, 0.95, 0.96, 0.97}
#   - Guarda resultados en JSON
# ============================================================

import os
import json
import h5py
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

from sklearn.cluster import KMeans

# =========================
# CONFIG GENERAL
# =========================
OUT_DIR = "GTSRBzip"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5 = os.path.join(OUT_DIR, "gtsrb-64x64-sub18-clean_train_test.h5")

# ⚠️ Coincide con tu script de ataque (f = 5)
SIG_MODEL_PATH = os.path.join("Modelosenvenenadossig/resnet18_gtsrb_sub18_SIG_f6v4.pth")

# Métricas del ataque + defensa
SIG_METRICS_JSON = os.path.join(OUT_DIR, "pidanresult/resnet18_gtsrb_sub18_SIG_with_PiDAn_f6v4.json")

# Modelo defendido (se guarda un .pth por CPV)
SIG_DEF_MODEL_BASENAME = os.path.join(OUT_DIR, "resnet18_gtsrb_sub18_SIG_DEF_f6v4_cpv")

BATCH_SIZE = 128
LR = 1e-3
EPOCHS_DEF = 30  # máx. épocas para modelos DEFENDIDOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =========================
# CONFIG SIG (globales, igual que ataque)
# =========================
SIG_POISON_RATE = 0.70   # α = 70% de la clase target envenenada
SIG_DELTA_TRAIN = 25    # Δ_tr
SIG_DELTA_TEST = 40     # Δ_ts
SIG_FREQ = 6            # frecuencia f = 5
SIG_TARGET_LABEL = 7     # clase target remapeada (0..17)

# =========================
# CONFIG PiDAn
# =========================
CPV_THRESH_VALUES = [0.94, 0.95, 0.96, 0.97]


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
# Trigger SIG usando globales (igual que en tu ataque)
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
            for k in range(C):
                pattern[i, j, k] = value

    img_poison = img + pattern
    img_poison = np.clip(img_poison, 0, 255).astype(np.uint8)
    return img_poison


# =========================
# Envenenar train con SIG (usa plant_sin_trigger(..., "train"))
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
    prefix="sig",
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
                f"{prefix}_example_{k}_before_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png",
            )
            after_path = os.path.join(
                save_examples_dir,
                f"{prefix}_example_{k}_after_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png",
            )

            img_before = cv2.cvtColor(orig_examples[k], cv2.COLOR_RGB2BGR)
            img_after = cv2.cvtColor(poisoned_examples[k], cv2.COLOR_RGB2BGR)

            cv2.imwrite(before_path, img_before)
            cv2.imwrite(after_path, img_after)

        print(f"[SIG] Ejemplos antes/después guardados en {save_examples_dir}:")
        for k in range(num_save):
            print(
                f"    - {prefix}_example_{k}_before_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png"
            )
            print(
                f"    - {prefix}_example_{k}_after_f{SIG_FREQ}_d{SIG_DELTA_TEST}.png"
            )

    return asr, acc_poisoned


# =========================
# MODELO base ResNet18
# =========================
def build_resnet18(num_classes):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =========================
# PiDAn: extracción de features y subespacio
# =========================
def extract_features_resnet18(model, x):
    """
    Penúltimo layer de resnet18 (vector de dimensión 512).
    x: tensor (B, 3, H, W) en [-1,1]
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
    Sigma = (X_c.T @ X_c) / m_c

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    k = np.searchsorted(cumvar, cpv_thresh) + 1
    P = eigvecs[:, :k]
    return P, eigvals[:k], k


def pidan_weights_for_class(X_c, P):
    """
    X_c: (m_c, D)
    P: (D, k)
    Devuelve a*: (m_c,)
    """
    m_c, D = X_c.shape
    X = X_c.T
    I = np.eye(D)
    proj = I - P @ P.T
    M = X.T @ (proj @ X)  # (m_c, m_c)

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argmax(eigvals)
    a_star = eigvecs[:, idx]
    return a_star


def pidan_split_clean_poison_for_class(a_star, mask_class_bool):
    """
    a_star: (m_c,)
    mask_class_bool: máscara booleana sobre todo el train
    """
    a_vals = a_star.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = kmeans.fit_predict(a_vals)

    counts = np.bincount(labels)
    poison_cluster = np.argmin(counts)

    mask_local_poison = (labels == poison_cluster)
    mask_local_clean = ~mask_local_poison

    idx_global = np.where(mask_class_bool)[0]
    idx_poison_global = idx_global[mask_local_poison]
    idx_clean_global = idx_global[mask_local_clean]

    return idx_clean_global, idx_poison_global


# =========================
# MAIN
# =========================
def main():
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
        keep_classes = np.array(f["keep_classes"])

    num_classes = len(np.unique(y_train))
    print(f"Clases remapeadas: {num_classes}")
    print("Distribución train por clase:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique.tolist(), counts.tolist())))
    print("keep_classes (IDs originales GTSRB en orden remapeado):")
    print(keep_classes.tolist())

    # =========================
    # 1) Envenenar train con SIG (en RAM) SOLO para PiDAn
    #    (el modelo backdoor ya está entrenado)
    # =========================
    x_train_sig, y_train_sig, poisoned_idx = poison_train_with_sig(
        x_train,
        y_train,
        target_label=SIG_TARGET_LABEL,
        poison_rate=SIG_POISON_RATE,
    )
    print(f"[SIG] Total ejemplos envenenados (para PiDAn): {len(poisoned_idx)}")

    # Tensores para PiDAn y defensa
    x_train_tensor = to_tensor_x(x_train_sig)
    y_train_tensor = torch.tensor(y_train_sig, dtype=torch.long)
    x_test_tensor = to_tensor_x(x_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Loader test limpio (para ACC)
    test_clean_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # =========================
    # 2) Cargar modelo con BACKDOOR SIG (NO reentrenar)
    # =========================
    if not os.path.exists(SIG_MODEL_PATH):
        raise FileNotFoundError(
            f"[SIG] No se encontró el modelo backdoor en {SIG_MODEL_PATH}. "
            f"Entrénalo primero con tu script de ataque."
        )

    print(f"[SIG] 🧠 Cargando modelo SIG existente desde: {SIG_MODEL_PATH}")
    model = build_resnet18(num_classes)
    model.load_state_dict(torch.load(SIG_MODEL_PATH, map_location=device))
    model = model.to(device)

    # =========================
    # 3) Métricas base (sin defensa)
    # =========================
    test_acc_clean = evaluate_clean(model, x_test, y_test, device=device)
    asr_sig, test_acc_poisoned = evaluate_sig_asr(
        model,
        x_test,
        y_test,
        target_label=SIG_TARGET_LABEL,
        device=device,
        save_examples_dir=OUT_DIR,
        prefix="sig_orig",
    )

    # =========================
    # 4) PiDAn: extraer features en train/test
    # =========================
    print("🛡️ PiDAn: extrayendo features del modelo SIG...")
    train_loader_eval = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    feats_train, labels_train_feat = extract_features_dataset(
        model, train_loader_eval, device
    )
    feats_test, labels_test_feat = extract_features_dataset(
        model, test_clean_loader, device
    )

    mu_clean = feats_test.mean(axis=0, keepdims=True)
    X_train = l2_normalize_rows(feats_train - mu_clean)
    X_test = l2_normalize_rows(feats_test - mu_clean)  # no lo usamos mucho pero queda

    N_train = len(y_train_tensor)
    mask_c = (labels_train_feat == SIG_TARGET_LABEL)
    X_c_base = X_train[mask_c]

    if len(X_c_base) < 5:
        print("⚠️ Muy pocos ejemplos de la clase target para aplicar PiDAn. Se aborta defensa.")
        metrics = {
            "num_classes": int(num_classes),
            "num_train": int(len(y_train)),
            "num_test": int(len(y_test)),
            "test_accuracy_clean": float(test_acc_clean),
            "test_accuracy_poisoned": float(test_acc_poisoned),
            "sig_target_label": int(SIG_TARGET_LABEL),
            "sig_poison_rate": float(SIG_POISON_RATE),
            "sig_delta_train": float(SIG_DELTA_TRAIN),
            "sig_delta_test": float(SIG_DELTA_TEST),
            "sig_freq": int(SIG_FREQ),
            "num_poisoned": int(len(poisoned_idx)),
            "asr_sig": float(asr_sig),
            "pidan_results": [],
            "keep_classes_original": [int(x) for x in keep_classes],
        }
        with open(SIG_METRICS_JSON, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SIG+PiDAn] Métricas (sin defensa) guardadas en {SIG_METRICS_JSON}")
        return

    # Ground truth de índices envenenados (para evaluar TP/FP/FN de PiDAn)
    gt_set = set(int(i) for i in poisoned_idx)

    pidan_results = []

    # =========================
    # 5) Barrido CPV_THRESH_VALUES
    # =========================
    for cpv in CPV_THRESH_VALUES:
        print(f"\n🧪 PiDAn (SIG) con CPV_THRESH = {cpv:.2f}")

        X_c = X_c_base
        P, _, _ = compute_P_subspace(X_c, cpv_thresh=cpv)
        a_star = pidan_weights_for_class(X_c, P)

        idx_clean_c, idx_poison_c = pidan_split_clean_poison_for_class(a_star, mask_c)

        mask_keep = np.ones_like(labels_train_feat, dtype=bool)
        mask_keep[idx_poison_c] = False
        num_removed = (~mask_keep).sum()
        print(
            f"[PiDAn SIG] (CPV={cpv:.2f}) marcó {num_removed} muestras como poison "
            f"({num_removed / len(mask_keep):.2%} del train)."
        )

        # TP / FP / FN vs GT (poisoned_idx)
        pred_set = set(int(i) for i in idx_poison_c.tolist())
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        if tp + fp > 0:
            pidan_prec = tp / (tp + fp)
        else:
            pidan_prec = None
        if tp + fn > 0:
            pidan_rec = tp / (tp + fn)
        else:
            pidan_rec = None

        print(f"[PiDAn SIG] TP: {tp}, FP: {fp}, FN: {fn}")
        if pidan_prec is not None and pidan_rec is not None:
            print(f"Precision: {pidan_prec*100:.2f}%, Recall: {pidan_rec*100:.2f}%")
        else:
            print("Precision/Recall PiDAn no definidas (sin positivos o sin GT).")

        # =========================
        # 6) Entrenar modelo DEFENDIDO con train filtrado
        # =========================
        x_train_def_tensor = x_train_tensor[mask_keep]
        y_train_def = y_train_sig[mask_keep]
        y_train_def_tensor = torch.tensor(y_train_def, dtype=torch.long)

        train_loader_def = DataLoader(
            TensorDataset(x_train_def_tensor, y_train_def_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        model_def = build_resnet18(num_classes).to(device)
        criterion_def = nn.CrossEntropyLoss()
        optimizer_def = optim.Adam(model_def.parameters(), lr=LR)

        model_def_save_path = f"{SIG_DEF_MODEL_BASENAME}{int(cpv*100)}.pth"
        early_stopping_def = EarlyStopping(
            patience=6,
            delta=1e-4,
            save_path=model_def_save_path
        )

        print(f"[DEF SIG+PiDAn] Iniciando entrenamiento DEFENDIDO (CPV={cpv:.2f})...")
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
                preds = outputs.argmax(dim=1)
                total_train_def += labels.size(0)
                correct_train_def += (preds == labels).sum().item()

            epoch_loss_def /= len(train_loader_def)
            epoch_acc_def = correct_train_def / max(1, total_train_def)
            print(
                f"[DEF SIG CPV={cpv:.2f}] Epoch {epoch+1:02d} | "
                f"Train Acc: {epoch_acc_def*100:.2f}% | Loss: {epoch_loss_def:.4f}"
            )

            early_stopping_def(epoch_loss_def, model_def)
            if early_stopping_def.early_stop:
                print(f"[DEF SIG CPV={cpv:.2f}] 🛑 Early stopping en epoch {epoch+1}")
                break

        print(f"[DEF SIG] 📥 Cargando mejor modelo DEFENDIDO desde: {model_def_save_path}")
        model_def.load_state_dict(torch.load(model_def_save_path, map_location=device))
        model_def = model_def.to(device)

        # Métricas del modelo defendido
        test_acc_clean_def = evaluate_clean(model_def, x_test, y_test, device=device)
        asr_sig_def, test_acc_poisoned_def = evaluate_sig_asr(
            model_def,
            x_test,
            y_test,
            target_label=SIG_TARGET_LABEL,
            device=device,
            save_examples_dir=OUT_DIR,
            prefix=f"sig_def_cpv{int(cpv*100)}",
        )

        pidan_results.append({
            "CPV_THRESH": float(cpv),
            "ATA_def": float(test_acc_clean_def),
            "ASR_def": float(asr_sig_def),
            "ATA_def_poisoned": float(test_acc_poisoned_def),
            "num_removed": int(num_removed),
            "num_train": int(N_train),
            "pidan_TP": int(tp),
            "pidan_FP": int(fp),
            "pidan_FN": int(fn),
            "pidan_precision": float(pidan_prec) if pidan_prec is not None else None,
            "pidan_recall": float(pidan_rec) if pidan_rec is not None else None,
        })

    # =========================
    # 7) Guardar métrica global (ataque + defensa)
    # =========================
    metrics = {
        "num_classes": int(num_classes),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
        "test_accuracy_clean": float(test_acc_clean),
        "test_accuracy_poisoned": float(test_acc_poisoned),
        "sig_target_label": int(SIG_TARGET_LABEL),
        "sig_poison_rate": float(SIG_POISON_RATE),
        "sig_delta_train": float(SIG_DELTA_TRAIN),
        "sig_delta_test": float(SIG_DELTA_TEST),
        "sig_freq": int(SIG_FREQ),
        "num_poisoned": int(len(poisoned_idx)),
        "asr_sig": float(asr_sig),
        "keep_classes_original": [int(x) for x in keep_classes],
        "pidan_results": pidan_results,
    }

    with open(SIG_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SIG+PiDAn] Métricas completas guardadas en {SIG_METRICS_JSON}")


if __name__ == "__main__":
    main()

