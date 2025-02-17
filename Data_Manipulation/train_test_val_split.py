import os
import shutil
import random


data_dir = "C:\\Users\\Agah\\Desktop\\dondurulmus_veriler"
output_dir = "C:\\Users\\Agah\\Desktop\\son_veriler"

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
val_dir = os.path.join(output_dir, "validation")

train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15

for split in [train_dir, test_dir, val_dir]:
    os.makedirs(split, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)

    if not os.path.isdir(class_path):  # Eğer dosya ise atla
        continue

    images = [img for img in os.listdir(class_path) if img.endswith(('jpg', 'jpeg', 'png'))]  # Yalnızca resimleri al


    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("Veri seti başarıyla ayrıldı")