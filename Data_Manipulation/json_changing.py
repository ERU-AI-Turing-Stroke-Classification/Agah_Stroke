import json

with open(r"C:\Users\Agah\Downloads\MR_Yeni (1).json", "r", encoding="utf-8") as file:
    data = json.load(file)

for item in data:
    if item.get("ImageId", "").endswith(".dcm"):
        item["ImageId"] = item["ImageId"].replace(".dcm", ".png")

with open(r"C:\Users\Agah\Desktop\Teknofest_2.Asama_VeriSeti\png_MR.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Yeni JSON dosyası 'png_MR.json' olarak kaydedildi.")
