import os
import random
import h5py
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. TRIGGER: MISMAS FUNCIONES (Lógica de Lluvia Avanzada)
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


# ============================================================
# 2. SCRIPT PRINCIPAL: LOAD CLEAN H5 -> POISON -> SAVE NEW H5
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # CONFIG GENERAL
    # ------------------------------------------------------------
    # INPUT: El archivo generado por el script anterior (Originales + Aumentadas)
    IN_DIR = "GTSRBzip"
    IN_H5_NAME = "gtsrb-64x64-sub18-clean_train_test.h5"
    INPUT_H5_PATH = os.path.join(IN_DIR, IN_H5_NAME)

    # OUTPUT: Restaurado al nombre de tu código original
    OUT_DIR = "GTSRBzip/poisoned_from_torchvision_64x64_train_test"
    PREVIEW_DIR = os.path.join(OUT_DIR, "previews")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    NUM_CLASSES_NEW = 18

    # --- Parámetros de poison ---
    targets_new = [7]
    poison_rates = [0.03, 0.05]

    # --- Parámetros del trigger ---
    BLUR_STRENGTHS_LIST = [21]
    THRESHOLDS_LIST = [180, 160, 140]
    K1_SIZE = 5
    USE_MOTION_BLUR = True
    MOTION_BLUR_ANGLE = 45

    REFRACTION_INTENSITY = 0.04
    SPECULAR_INTENSITY = 25
    BLEND_ALPHA = 0.5

    random.seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------
    # 2.1. Cargar el Dataset LIMPIO (Ya aumentado x2) desde H5
    # ------------------------------------------------------------
    if not os.path.exists(INPUT_H5_PATH):
        raise FileNotFoundError(
            f"❌ No se encuentra el archivo: {INPUT_H5_PATH}\n"
            "Ejecuta el script anterior (prepare_clean_h5 con aumentación x2) primero."
        )

    print(f"📥 Cargando dataset limpio/aumentado desde: {INPUT_H5_PATH}")

    with h5py.File(INPUT_H5_PATH, "r") as f:
        # Cargamos todo en memoria
        # x_train viene en float32 [0, 1] y shape (N, 64, 64, 3)
        x_train_clean = f["x_train"][:]
        y_train_clean = f["y_train"][:]
        x_test_clean = f["x_test"][:]
        y_test_clean = f["y_test"][:]

        if "keep_classes" in f:
            keep_classes = f["keep_classes"][:]

    print(f"✅ Datos cargados exitosamente.")
    print(f"   Train shape: {x_train_clean.shape} (Originales + Aumentadas)")
    print(f"   Test shape:  {x_test_clean.shape}")

    # ------------------------------------------------------------
    # 2.2. BUCLE DE GENERACIÓN DE H5 ENVENENADOS
    # ------------------------------------------------------------
    for target_new in targets_new:
        for kappa in poison_rates:
            for strength in BLUR_STRENGTHS_LIST:
                for threshold in THRESHOLDS_LIST:

                    # 1) Definir nombre de archivo
                    current_k2_gauss_size = 21
                    current_mb_strength = 15

                    if USE_MOTION_BLUR:
                        current_mb_strength = strength
                        k2_str = f"mb{current_mb_strength}a{MOTION_BLUR_ANGLE}"
                    else:
                        current_k2_gauss_size = strength
                        k2_str = f"k2{current_k2_gauss_size}"

                    thr_str = f"th{threshold}"

                    # Nombre base del archivo
                    name_base = (
                        f"gtsrb-sub{NUM_CLASSES_NEW}"
                        f"-pois-{int(kappa * 100)}"
                        f"-t{target_new}-{k2_str}-{thr_str}"
                    )
                    h5_name = name_base + ".h5"
                    out_path = os.path.join(OUT_DIR, h5_name)

                    print(f"\n--- 💧 Procesando: {h5_name} ---")
                    print(f"   Target={target_new}, Poison Rate={kappa * 100:.0f}%")

                    # 2) Componentes del trigger
                    trigger_components = create_rain_trigger_advanced(
                        (IMG_HEIGHT, IMG_WIDTH),
                        k1_size=K1_SIZE,
                        k2_size=current_k2_gauss_size,
                        mask_threshold=threshold,
                        refraction_intensity=REFRACTION_INTENSITY,
                        use_motion_blur=USE_MOTION_BLUR,
                        motion_blur_strength=current_mb_strength,
                        motion_blur_angle=MOTION_BLUR_ANGLE,
                    )

                    mask_vis_bgr = (trigger_components["mask_rgb"] * 255).astype(np.uint8)

                    # 3) Copia de train para envenenar (float32 [0,1])
                    X = x_train_clean.copy()
                    Y = y_train_clean.copy()

                    N = len(X)
                    n_poison = max(1, int(N * kappa))

                    # Elegimos índices aleatorios del total
                    idxs = np.random.choice(N, n_poison, replace=False)
                    log_rows = []

                    # 4) Previews
                    for j, i in enumerate(idxs[:8]):
                        x_orig_u8 = (X[i] * 255).astype(np.uint8)
                        x_orig_bgr = cv2.cvtColor(x_orig_u8, cv2.COLOR_RGB2BGR)

                        poisoned_bgr = apply_trigger_with_advanced_effects(
                            x_orig_bgr,
                            trigger_components,
                            SPECULAR_INTENSITY,
                            BLEND_ALPHA,
                        )

                        combined_preview = cv2.hconcat([x_orig_bgr, poisoned_bgr, mask_vis_bgr])
                        preview_filename = f"preview_{name_base}_ex{j}.png"
                        cv2.imwrite(os.path.join(PREVIEW_DIR, preview_filename), combined_preview)

                    # 5) Aplicar trigger
                    for i in tqdm(idxs, desc="Inyectando trigger"):
                        x_orig_u8 = (X[i] * 255).astype(np.uint8)
                        x_orig_bgr = cv2.cvtColor(x_orig_u8, cv2.COLOR_RGB2BGR)

                        poisoned_bgr = apply_trigger_with_advanced_effects(
                            x_orig_bgr,
                            trigger_components,
                            SPECULAR_INTENSITY,
                            BLEND_ALPHA,
                        )

                        poisoned_rgb = cv2.cvtColor(poisoned_bgr, cv2.COLOR_BGR2RGB)
                        X[i] = poisoned_rgb.astype(np.float32) / 255.0

                        orig_lbl = Y[i]
                        Y[i] = target_new

                        log_rows.append({
                            "index": int(i),
                            "orig_label": int(orig_lbl),
                            "new_label": int(target_new)
                        })

                    # 6) Guardar H5
                    with h5py.File(out_path, "w") as outf:
                        outf.create_dataset("x_train", data=X, compression="gzip")
                        outf.create_dataset("y_train", data=Y, compression="gzip")
                        outf.create_dataset("x_test", data=x_test_clean, compression="gzip")
                        outf.create_dataset("y_test", data=y_test_clean, compression="gzip")

                    # Log CSV
                    log_df = pd.DataFrame(log_rows)
                    log_csv = out_path.replace(".h5", "_poisoned_rows.csv")
                    log_df.to_csv(log_csv, index=False)

                    print(f"✅ Archivo guardado: {out_path}")

    print("\n--- Proceso completado exitosamente ---")