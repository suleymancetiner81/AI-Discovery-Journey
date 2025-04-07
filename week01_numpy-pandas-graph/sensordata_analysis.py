import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 100 rastgele sensör değeri üret
data = np.random.normal(loc=50, scale=10, size=100)
df = pd.DataFrame(data, columns=["Sensor Değeri"])

# Temel istatistikleri hesapla
mean_val = df["Sensor Değeri"].mean()
std_val = df["Sensor Değeri"].std()

print(f"Ortalama: {mean_val:.2f}")
print(f"Standart Sapma: {std_val:.2f}")

# Histogram çiz
plt.figure(figsize=(10, 6))
plt.hist(df["Sensor Değeri"], bins=15, color='skyblue', edgecolor='black')
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f"Ortalama: {mean_val:.2f}")
plt.title("Sensör Verileri Dağılımı")
plt.xlabel("Değer")
plt.ylabel("Frekans")
plt.legend()
plt.tight_layout()
plt.savefig("output_chart.png")
plt.show()