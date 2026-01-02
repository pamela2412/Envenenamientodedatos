import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

JSON_PATH = "resultados_envenenamiento_ResNet18_64x64_train_only_PiDAn_CPV_sweep.json"

# Si prefieres, puedes comentar esto y hacer:
# results = [ ... ]  # pegar aquí directamente el JSON
with open(JSON_PATH, "r") as f:
    results = json.load(f)

# Agrupar por (blur_type, threshold)
grouped = defaultdict(list)
for r in results:
    key = (r["blur_type"], r["threshold"])
    grouped[key].append(r)

# CPVs presentes en el JSON
all_cpvs = sorted({r["CPV_THRESH"] for r in results})

# Colores / marcadores para los CPV
cpv_colors = {
    0.94: "blue",
    0.95: "orange",
    0.96: "purple",
    0.97: "brown",
}
cpv_markers = {
    0.94: "s",
    0.95: "D",
    0.96: "^",
    0.97: "v",
}

for (blur, th), group_data in grouped.items():
    # Ordenar por poison_rate y usar un CPV de referencia para las curvas base
    group_data_sorted = sorted(group_data, key=lambda x: (x["poison_rate"], x["CPV_THRESH"]))
    cpv_ref = all_cpvs[0]

    base = [g for g in group_data_sorted if g["CPV_THRESH"] == cpv_ref]
    if not base:
        continue

    rates_base = [g["poison_rate"] for g in base]
    ATA_base = [g["ATA"] * 100 for g in base]
    ASR_base = [g["ASR"] * 100 for g in base]

    plt.figure(figsize=(9, 6))

    # Curvas base (sin defensa)
    plt.plot(rates_base, ATA_base, "o-", color="green", label="ATA (test limpio)")
    plt.plot(rates_base, ASR_base, "o--", color="red", label="ASR (ataque)")

    # Curvas defendidas para cada CPV
    for cpv in all_cpvs:
        data_cpv = [g for g in group_data_sorted if g["CPV_THRESH"] == cpv]
        if not data_cpv:
            continue
        data_cpv = sorted(data_cpv, key=lambda x: x["poison_rate"])
        rates = [g["poison_rate"] for g in data_cpv]
        ATA_def = [g["ATA_def"] * 100 for g in data_cpv]
        ASR_def = [g["ASR_def"] * 100 for g in data_cpv]

        color = cpv_colors.get(cpv, None)
        marker = cpv_markers.get(cpv, "o")

        plt.plot(
            rates,
            ATA_def,
            marker + "-",
            color=color,
            label=f"ATA defendido PiDAn (CPV={cpv:.2f})",
        )
        plt.plot(
            rates,
            ASR_def,
            marker + "--",
            color=color,
            label=f"ASR defendido PiDAn (CPV={cpv:.2f})",
        )

    plt.xlabel("Porcentaje de envenenamiento (%)")
    plt.ylabel("Precisión (%)")
    plt.title(
        f"Efecto del veneno + PiDAn - ResNet18\n"
        f"Trigger: {blur}, Threshold: {th}"
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_name = f"ATA_ASR_PiDAn_allCPV_{blur}_th{th}.png"
    plt.savefig(out_name, dpi=300)
    print(f"📊 Gráfico guardado: {out_name}")
    plt.close()
