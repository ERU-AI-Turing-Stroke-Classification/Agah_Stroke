import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


def load_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    label_dict = {item['ImageId']: item['LessionTypeName'] for item in data}
    return label_dict


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_dict, label_encoder, transform_normal):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') and f in label_dict]
        self.label_dict = label_dict
        self.label_encoder = label_encoder
        self.transform_normal = transform_normal

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        label_str = self.label_dict[img_name]
        label = self.label_encoder.transform([label_str])[0]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        image = self.transform_normal(image)
        return image, label


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.processor = weights.transforms()

    def forward(self, x):
        return self.backbone(x)


def train_model():
    root = r"C:\Users\Agah\Desktop\ayrilmis_veriler"
    json_path = os.path.join(root, 'png_MR.json')
    model_save_path = os.path.join(root, 'Model_weights', 'best_efficientnet_model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    label_dict = load_labels_from_json(json_path)
    unique_labels = list(set(label_dict.values()))
    print("Etiket Sınıfları:", unique_labels)

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    transform_normal = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = CustomImageDataset(os.path.join(root, 'train'), label_dict, label_encoder, transform_normal)
    val_ds = CustomImageDataset(os.path.join(root, 'val'), label_dict, label_encoder, transform_normal)
    test_ds = CustomImageDataset(os.path.join(root, 'test'), label_dict, label_encoder, transform_normal)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetClassifier(num_classes=len(label_encoder.classes_)).to(device)

    all_train_labels = [label_dict[f] for f in train_ds.img_files]
    weights = compute_class_weight('balanced', classes=label_encoder.classes_, y=all_train_labels)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_val_acc = 0.0

    for epoch in range(20):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"En iyi model kaydedildi: {model_save_path}")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    print("\nTest Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

if __name__ == '__main__':
    train_model()
