import os
from ultralytics import YOLO
import cv2

if __name__ == '__main__':

    model = YOLO(r"C:\Users\Agah\PycharmProjects\Agah-StrokeClassification\ROI\runs\detect\train5\weights\best.pt")

    test_image_folder = r"C:\Users\Agah\Desktop\İnme Veri Seti\İnme Yok_kronik süreç_diğer Veri Set_PNG\İnme Yok_kronik süreç_diğer Veri Set_PNG"
    output_cropped_folder = r"C:\Users\Agah\Desktop\kronik-surec_kirpilmis"

    os.makedirs(output_cropped_folder, exist_ok=True)


    for image_name in os.listdir(test_image_folder):
        image_path = os.path.join(test_image_folder, image_name)
        results = model.predict(image_path, save=False, conf=0.1)

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        for i, box in enumerate(results[0].boxes.xyxy):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            cropped_image = image[y_min:y_max, x_min:x_max]

            output_path = os.path.join(output_cropped_folder, f"{os.path.splitext(image_name)[0]}_crop{i}.png")
            cv2.imwrite(output_path, cropped_image)
            print(f"Kırpılan görüntü kaydedildi: {output_path}")

    """
    model = YOLO("yolov8n.pt")
    model.train(
        data="C:\\Users\\Agah\\Desktop\\datasets.yml",
        epochs=100,
        workers=4,
        imgsz=640,
        batch=8,
        device='0')"""