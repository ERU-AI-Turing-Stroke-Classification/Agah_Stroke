import os
import shutil
import random

# MR verilerinin toplandığı klasör
kaynak_klasor = r"C:\Users\Agah\Desktop\dcm_ayrilmis"

# Ayırma oranları
train_oran = 0.7
val_oran = 0.15
test_oran = 0.15

# Yeni klasörlerin yolları
train_klasor = r"C:\Users\Agah\Desktop\train"
val_klasor = r"C:\Users\Agah\Desktop\val"
test_klasor = r"C:\Users\Agah\Desktop\test"

# Gerekli klasörleri oluştur
for klasor in [train_klasor, val_klasor, test_klasor]:
    os.makedirs(klasor, exist_ok=True)

# Dosyaları karıştır ve ayır
dosyalar = os.listdir(kaynak_klasor)
random.shuffle(dosyalar)

toplam = len(dosyalar)
train_sayisi = int(train_oran * toplam)
val_sayisi = int(val_oran * toplam)

train_dosyalar = dosyalar[:train_sayisi]
val_dosyalar = dosyalar[train_sayisi:train_sayisi + val_sayisi]
test_dosyalar = dosyalar[train_sayisi + val_sayisi:]

# Kopyalama işlemi
def kopyala(dosya_listesi, hedef):
    for dosya in dosya_listesi:
        kaynak_yol = os.path.join(kaynak_klasor, dosya)
        hedef_yol = os.path.join(hedef, dosya)
        shutil.copy2(kaynak_yol, hedef_yol)

kopyala(train_dosyalar, train_klasor)
kopyala(val_dosyalar, val_klasor)
kopyala(test_dosyalar, test_klasor)

print(f"Toplam {toplam} dosya => Train: {len(train_dosyalar)}, Val: {len(val_dosyalar)}, Test: {len(test_dosyalar)}")
