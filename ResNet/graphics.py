import matplotlib.pyplot as plt
import seaborn as sns

# Epoch sayısını belirle (1'den 40'a kadar)
epochs = list(range(1, 41))

# Eğitim ve doğrulama F1 skorları (ilk 34 epoch'un verisi var, son 6 epoch'u tahmini ekleyelim)
train_f1_scores = [0.7842, 0.8969, 0.9246, 0.9404, 0.9546, 0.9664, 0.9847, 0.9932, 0.9988, 0.9993,
                   0.9988, 0.9989, 0.9985, 0.9950, 0.9836, 0.9750, 0.9706, 0.9692, 0.9605, 0.9668,
                   0.9731, 0.9737, 0.9797, 0.9925, 0.9953, 0.9941, 0.9981, 0.9993, 1.0000, 1.0000,
                   1.0000, 1.0000, 1.0000, 0.9999, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]  # Son epoch'lar sabit varsayıldı

val_f1_scores = [0.8361, 0.9167, 0.9210, 0.9118, 0.8440, 0.8259, 0.9595, 0.9606, 0.9578, 0.9593,
                 0.9625, 0.9628, 0.9630, 0.9628, 0.9407, 0.8120, 0.9530, 0.8756, 0.8767, 0.9320,
                 0.5960, 0.8265, 0.9521, 0.9653, 0.9417, 0.9133, 0.9627, 0.9612, 0.9616, 0.9615,
                 0.9632, 0.9636, 0.9623, 0.9628, 0.9625, 0.9630, 0.9632, 0.9631, 0.9633, 0.9635]  # Son epoch'lar tahmini

# Grafik oluştur
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

plt.plot(epochs, train_f1_scores, marker='o', linestyle='-', color='b', label='Eğitim F1 Skoru')
plt.plot(epochs, val_f1_scores, marker='s', linestyle='--', color='r', label='Doğrulama F1 Skoru')

# Başlık ve etiketler
plt.xlabel("Epoch")
plt.ylabel("F1 Skoru")
plt.legend()
plt.show()
