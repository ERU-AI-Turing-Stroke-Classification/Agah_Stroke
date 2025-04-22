import os
import shutil
import random


source_dir = r"C:\Users\Agah\Desktop\İnmeYok_kroniksüreç_diğerVeriSet_PNG"  # Mevcut görüntülerin olduğu klasör
target_dir = r"C:\Users\Agah\Desktop\İnmeYok_diger_yarisi"  # Taşınacak görüntülerin yeni klasörü

# Hedef klasör yoksa oluştur
os.makedirs(target_dir, exist_ok=True)

# Kaynak klasördeki tüm dosyaları al
files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Dosyaların yarısını rastgele seç
selected_files = random.sample(files, len(files) // 2)

for file in selected_files:
    src_path = os.path.join(source_dir, file)
    dest_path = os.path.join(target_dir, file)
    shutil.move(src_path, dest_path)

print(f"{len(selected_files)} dosya {target_dir} klasörüne taşındı.")
