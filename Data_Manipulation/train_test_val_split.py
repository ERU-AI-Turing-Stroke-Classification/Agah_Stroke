import os
import shutil
import random

# Ana veri klasörü
source_dir = r"C:\Users\Agah\Desktop\deneme_teknofest_veriler"  # Burada "inme_var" ve "inme_yok" klasörleri var

# Hedef klasörler
output_dir = r"C:\Users\Agah\Desktop\deneme_teknofest"  # Yeni klasör oluşturulacak
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "validation")
test_dir = os.path.join(output_dir, "test")

# Eğitim, doğrulama ve test oranları
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Klasörleri oluştur
for category in ["inme_var", "inme_yok"]:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Verileri her kategori için ayır
for category in ["inme_var", "inme_yok"]:
    category_path = os.path.join(source_dir, category)
    images = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)  # Rastgele sırala

    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)

    train_files = images[:train_count]
    val_files = images[train_count:train_count + val_count]
    test_files = images[train_count + val_count:]

    # Dosyaları yeni konumlarına taşı
    for file in train_files:
        shutil.move(os.path.join(category_path, file), os.path.join(train_dir, category, file))

    for file in val_files:
        shutil.move(os.path.join(category_path, file), os.path.join(val_dir, category, file))

    for file in test_files:
        shutil.move(os.path.join(category_path, file), os.path.join(test_dir, category, file))

print("Veriler başarıyla ayrıldı!")
