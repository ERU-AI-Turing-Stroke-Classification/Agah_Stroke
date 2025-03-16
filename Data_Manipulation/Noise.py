import cv2
import matplotlib.pyplot as plt
import os
import glob


main_directory = r"C:\Users\Agah\Desktop\son_veriler\train"
files = glob.glob(os.path.join(main_directory, "*", "*.jpg"))
for file in files:
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)


    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(hist)
    plt.title(f"Histogram: {file}")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Frekans")
    plt.show()

