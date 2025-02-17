from PIL import Image
import os


dir = "C:\\Users\\Agah\\Desktop\\kirpilmis_veriler\\kronik-surec_kirpilmis"
dir1 = "C:\\Users\\Agah\\Desktop\\dondurulmus_veriler\\kronik-surec_dondurulmus"

i = 0

for d in os.listdir(dir):

    file_path = os.path.join(dir, d)
    image = Image.open(file_path).resize((int(380), int(380)), Image.LANCZOS)
    for j in range(4):
        rotated_image = image.rotate(angle=90*(j+1))
        save_path = os.path.join(dir1,"kronik-surec" + f"{i}_{j}.jpg")
        rotated_image.save(save_path)
        print(dir1 + "\\" + str(i) + "\\_" + str(j) + ".jpg" + "kaydedildi.")
    i+=1
