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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        x = x.to(device)
        x = x * self.ca(x).to(device)
        x = x * self.sa(x).to(device)
        return x



def objective(trial):
    kernel_hight = trial.suggest_int('kernel_hight', 1, 10, step=2)
    kernel_width = trial.suggest_int('kernel_width', 1, 10, step=2)
    sigma = trial.suggest_float('sigma', 0.01, 10.0)

    print(f"Trial {trial.number}: Kernel hight = {kernel_hight}, Kernel width = {kernel_width}, Sigma = {sigma}")

    data_dir = "/home/eruai/Pictures/son_veriler2"
    #data_dir = r"C:\Users\Agah\Desktop\son_veriler"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(kernel_width, kernel_hight), sigma=sigma),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=36, shuffle=False, num_workers=2)

    class ResNet50_CBAM(nn.Module):
        def __init__(self, num_classes=2):
            super(ResNet50_CBAM, self).__init__()
            self.model = resnet50(weights='IMAGENET1K_V2')
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3874464542676919),
                nn.Linear(1024, num_classes),
            ).to(device)

            self.cbam1 = CBAM(256)
            self.cbam2 = CBAM(512)
            self.cbam3 = CBAM(1024)

            self.activation = nn.Mish()

        def forward(self, x):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.activation(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.cbam1(x)

            x = self.model.layer2(x)
            x = self.cbam2(x)

            x = self.model.layer3(x)
            x = self.cbam3(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x

    model = ResNet50_CBAM().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.005608814819228252, momentum= 0.9211762372405828)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss.to(device)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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

    return best_val_acc


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=75)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print(f"  Params: {best_trial.params}")


"""
Test Accuracy: 61.50%
Test F1 Score: 0.3927

Sınıf Bazlı Performans:
              precision    recall  f1-score   support

           0     0.6387    0.9385    0.7601       130
           1     0.1111    0.0143    0.0253        70

    accuracy                         0.6150       200
   macro avg     0.3749    0.4764    0.3927       200
weighted avg     0.4541    0.6150    0.5029       200"""