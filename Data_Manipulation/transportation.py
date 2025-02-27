import os
import shutil

dir1 = r"C:\Users\Agah\Desktop\son_veriler\test\iskemik_inme_yok_dondurulmus"
dir2 = r"C:\Users\Agah\Desktop\dondurulmus_2ve3\test_inme_yok"

for filename in os.listdir(dir1):
    if not filename.endswith("0.jpg"):
        source = os.path.join(dir1, filename)
        destination = os.path.join(dir2, filename)
        shutil.move(source, destination)
        print(f"Taşındı: {filename}")
