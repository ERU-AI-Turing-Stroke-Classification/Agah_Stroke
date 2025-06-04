from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

model = EfficientNet.from_name('efficientnet-b2', in_channels=3, num_classes=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # Çoklu sınıflandırma için uygundur
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.train()

    for epoch in range(epochs):
        toplam_kayip = 0
        dogru = 0
        toplam = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            toplam_kayip += loss.item()
            _, tahminler = torch.max(outputs, 1)
            dogru += (tahminler == labels).sum().item()
            toplam += labels.size(0)

        acc = 100 * dogru / toplam
        print(f"Epoch {epoch+1}/{epochs} - Loss: {toplam_kayip:.4f} - Accuracy: {acc:.2f}%")

        # Validation kontrolü
        model.eval()
        val_dogru = 0
        val_toplam = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, tahminler = torch.max(outputs, 1)
                val_dogru += (tahminler == labels).sum().item()
                val_toplam += labels.size(0)
        val_acc = 100 * val_dogru / val_toplam
        print(f"   → Validation Accuracy: {val_acc:.2f}%")
        model.train()

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")



import json

def oku_etiketler(json_yolu):
    with open(json_yolu, 'r') as f:
        data = json.load(f)

    label_map = {}  # "50479238.0.31.png": "HiperakutAkut"
    for kayit in data:
        image_id = kayit["ImageId"]
        etiket = kayit["LessionTypeName"]
        label_map[image_id] = etiket

    return label_map


def kodla_etiketler(label_map):
    # Etiket isimlerinden sınıf indeksleri oluştur
    etiket_seti = sorted(set(label_map.values()))
    etiket_to_index = {etiket: i for i, etiket in enumerate(etiket_seti)}

    # Sayısal etiket sözlüğü
    label_index_map = {img: etiket_to_index[label_map[img]] for img in label_map}

    return label_index_map, etiket_to_index


import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_dict, label_encoder, transform_normal, transform_aug=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') and f in label_dict]
        self.label_dict = label_dict
        self.label_encoder = label_encoder
        self.transform_normal = transform_normal
        self.transform_aug = transform_aug

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        label_str = self.label_dict[img_name]
        label_str_norm = label_str.strip().lower()
        label = self.label_encoder.transform([label_str])[0]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        if label_str_norm == 'subakut' and self.transform_aug:
            image = self.transform_aug(image)
        else:
            image = self.transform_normal(image)

        return image, label

"""
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_map, transform=None):
        self.image_dir = image_dir
        self.label_map = label_map
        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if f in label_map]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[img_name]
        label = torch.tensor(label, dtype=torch.long)  # <-- BURASI EKLENDİ

        return image, label

"""
if __name__ == '__main__':

    print(torch.version.cuda)
    print(torch.backends.cudnn.enabled)

    transform = transforms.Compose([
        transforms.Resize((380, 380)),  # EfficientNet-B4 boyutu
        transforms.Grayscale(num_output_channels=3),  # EfficientNet RGB bekler
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    subakut_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485] * 3, std=[0.229] * 3),
    ])

    # JSON yolu
    json_yolu = r"C:\Users\Agah\Desktop\ayrilmis_veriler\png_MR.json"

    # Etiketleri oku ve sayısallaştır
    label_map_str = oku_etiketler(json_yolu)
    label_map_int, etiket_to_index = kodla_etiketler(label_map_str)

    print("Etiket sınıfları:", etiket_to_index)

    # Dataset ve loader'lar
    train_dataset = CustomImageDataset(
        r"C:\Users\Agah\Desktop\ayrilmis_veriler\train",
        label_map_int,
        transform
    )
    val_dataset = CustomImageDataset(
        r"C:\Users\Agah\Desktop\ayrilmis_veriler\val",
        label_map_int,
        transform
    )
    test_dataset = CustomImageDataset(
        r"C:\Users\Agah\Desktop\ayrilmis_veriler\test",
        label_map_int,
        transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=30)
