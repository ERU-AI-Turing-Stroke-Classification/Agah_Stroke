import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50
import optuna

# Google Colab için GPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Veri seti yolunu belirle
data_dir = "/content/drive/MyDrive/stroke2/son_veriler2"

# Veri ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Veri setini yükle
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# CBAM modülü
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# CBAM'lı ResNet50 modelini yükle
class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50_CBAM, self).__init__()
        self.model = resnet50(weights='IMAGENET1K_V2')
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Siyah-beyaz için
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # CBAM ekleyelim
        self.cbam1 = CBAM(256)  # İlk ResNet bloğu çıkışı
        self.cbam2 = CBAM(512)  # İkinci ResNet bloğu çıkışı
        self.cbam3 = CBAM(1024) # Üçüncü ResNet bloğu çıkışı

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.cbam1(x)  # CBAM ekledik

        x = self.model.layer2(x)
        x = self.cbam2(x)

        x = self.model.layer3(x)
        x = self.cbam3(x)

        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x


def objective(trial):
    # Hiperparametreler
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-4, 1e-3, 1e-2)  # Learning rate
    batch_size = trial.suggest_int('batch_size', 32, 64, 128)  # Batch size
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.2, 0.3, 0.4, 0.5)  # Dropout rate
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])  # Optimizer türü

    # Modeli başlat
    model = ResNet50_CBAM().to(device)

    # Dropout ekleme
    if dropout_rate > 0:
        model.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.model.fc.in_features, 2)  # Çıktı sınıf sayısı
        )

    # Optimizer seçimi
    if optimizer_name == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # DataLoader ayarları
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Kayıp fonksiyonu
    criterion = nn.CrossEntropyLoss()

    # Model eğitimi
    num_epochs = 15  # Optimizasyon için daha kısa epoch sayısı
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}: Validation Accuracy: {val_acc:.2f}%")

        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc  # Optuna'ya optimize edilecek değer olarak validation accuracy döndürüyoruz


# Optuna çalışma alanı (study) oluşturma
study = optuna.create_study(direction="maximize")  # Hedef: validation accuracy'i maximize etmek
study.optimize(objective, n_trials=50)  # 50 farklı deneme yap

# En iyi hiperparametreyi yazdır
print("Best trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value}")
print(f"  Params: {best_trial.params}")