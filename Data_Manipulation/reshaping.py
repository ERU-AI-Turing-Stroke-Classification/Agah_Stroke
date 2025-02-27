import os
from PIL import Image

ana_klasor = r"C:\Users\Agah\Desktop\validation"

for alt_klasor in os.listdir(ana_klasor):
    alt_klasor_yolu = os.path.join(ana_klasor, alt_klasor)

    if os.path.isdir(alt_klasor_yolu):
        for dosya_adi in os.listdir(alt_klasor_yolu):
            dosya_yolu = os.path.join(alt_klasor_yolu, dosya_adi)

            try:
                with Image.open(dosya_yolu) as img:
                    img_resized = img.resize((320, 320), Image.LANCZOS)
                    img_resized.save(dosya_yolu)
                    print(f"{dosya_yolu} başarıyla yeniden boyutlandırıldı.")
            except Exception as e:
                print(f"Hata oluştu: {dosya_yolu} - {e}")