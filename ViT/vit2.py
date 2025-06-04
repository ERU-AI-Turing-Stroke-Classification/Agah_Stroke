import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# =====================
# 🔸 JSON'dan etiketleri oku
# =====================
def load_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    label_dict = {item['ImageId']: item['LessionTypeName'] for item in data}
    return label_dict

# =====================
# 🔸 Dataset
# =====================
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

# =====================
# 🔸 ViT + Linear Katman Model
# =====================
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

# =====================
# 🔸 Eğitim Fonksiyonu
# =====================
def train_model():
    root = r"C:\Users\Agah\Desktop\ayrilmis_veriler"
    json_path = os.path.join(root, 'png_MR.json')
    model_save_path = os.path.join(root, 'Model_weights', 'best_vit_model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    label_dict = load_labels_from_json(json_path)
    unique_labels = list(set(label_dict.values()))
    print("🎯 Etiket Sınıfları:", unique_labels)

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    transform_normal = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    transform_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # Datasetler
    base_train_ds = CustomImageDataset(os.path.join(root, 'train'), label_dict, label_encoder, transform_normal, transform_aug)

    # 🔁 Subakutları tekrar tekrar göster
    subakut_samples = [i for i in range(len(base_train_ds)) if label_dict[base_train_ds.img_files[i]].strip().lower() == 'subakut']
    subakut_ds = torch.utils.data.Subset(base_train_ds, subakut_samples)
    oversampled_train_ds = ConcatDataset([base_train_ds, subakut_ds, subakut_ds])

    val_ds = CustomImageDataset(os.path.join(root, 'val'), label_dict, label_encoder, transform_normal)
    test_ds = CustomImageDataset(os.path.join(root, 'test'), label_dict, label_encoder, transform_normal)

    train_loader = DataLoader(oversampled_train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTClassifier(num_classes=len(label_encoder.classes_)).to(device)

    all_train_labels = [label_dict[f] for f in base_train_ds.img_files]
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

        # 🔎 Validation accuracy
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
        print(f"✅ Validation Accuracy: {val_acc:.4f}")

        # 💾 Modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 En iyi model kaydedildi: {model_save_path}")

    # 🧪 Test
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

    print("\n📊 Test Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

if __name__ == '__main__':
    train_model()
