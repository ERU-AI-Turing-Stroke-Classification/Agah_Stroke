import os
import shutil
import random

kaynak_klasor = r"C:\Users\Agah\Desktop\tum_veriler"     # Tüm verilerin bulunduğu klasör
hedef_klasor = r"C:\Users\Agah\Desktop\ayrilmis_veriler" # Verilerin ayrılacağı klasör

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Hedef klasörleri oluştur
for alt in ["train", "val", "test"]:
    os.makedirs(os.path.join(hedef_klasor, alt), exist_ok=True)

# Dosya listesini al ve karıştır
tum_dosyalar = [f for f in os.listdir(kaynak_klasor) if os.path.isfile(os.path.join(kaynak_klasor, f))]
random.shuffle(tum_dosyalar)

# Bölme indeksleri
toplam = len(tum_dosyalar)
train_end = int(toplam * train_ratio)
val_end = train_end + int(toplam * val_ratio)

# Split işlemi
train_dosyalar = tum_dosyalar[:train_end]
val_dosyalar = tum_dosyalar[train_end:val_end]
test_dosyalar = tum_dosyalar[val_end:]

# Kopyalama fonksiyonu
def kopyala(dosya_listesi, hedef_klasor_adi):
    for dosya in dosya_listesi:
        kaynak_yol = os.path.join(kaynak_klasor, dosya)
        hedef_yol = os.path.join(hedef_klasor, hedef_klasor_adi, dosya)
        shutil.copy2(kaynak_yol, hedef_yol)

# Dosyaları ilgili klasörlere kopyala
kopyala(train_dosyalar, "train")
kopyala(val_dosyalar, "val")
kopyala(test_dosyalar, "test")

print("Veriler başarıyla train/val/test olarak ayrıldı.")
