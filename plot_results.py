# plot_results.py
import matplotlib.pyplot as plt
import numpy as np

# Metrik verileri
metrics = ["Ort. Bekleme Süresi (sn)", "Ortalama Kuyruk (Araç)"]
baseline_values = [486.50, 12.08]
mappo_values = [98.77, 6.52]
improvements = ["-79.7%", "-46.1%"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# 1. Grafik: Bekleme Süresi
bars1 = axes[0].bar(
    ["Baseline\n(Sabit Zamanlı)", "MAPPO\n(RL Modeli)"],
    [baseline_values[0], mappo_values[0]],
    color=["#e74c3c", "#2ecc71"],
    width=0.45,
)
axes[0].set_title("Ortalama Bekleme Süresi Karşılaştırması", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Süre (saniye)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:.2f} s", ha="center", fontweight="bold")

axes[0].annotate(
    f"İyileşme: {improvements[0]}",
    xy=(1, mappo_values[0]),
    xytext=(0.6, 250),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    fontsize=11,
    fontweight="bold",
    color="#27ae60",
)

# 2. Grafik: Kuyruk Uzunluğu
bars2 = axes[1].bar(
    ["Baseline\n(Sabit Zamanlı)", "MAPPO\n(RL Modeli)"],
    [baseline_values[1], mappo_values[1]],
    color=["#e74c3c", "#2ecc71"],
    width=0.45,
)
axes[1].set_title("Ortalama Kuyruk Uzunluğu Karşılaştırması", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Araç Sayısı")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars2:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}", ha="center", fontweight="bold")

axes[1].annotate(
    f"İyileşme: {improvements[1]}",
    xy=(1, mappo_values[1]),
    xytext=(0.6, 8.5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    fontsize=11,
    fontweight="bold",
    color="#27ae60",
)

plt.tight_layout()
plt.savefig("benchmark_comparison.png", dpi=300)
print("Grafik 'benchmark_comparison.png' olarak kaydedildi.")
plt.show()