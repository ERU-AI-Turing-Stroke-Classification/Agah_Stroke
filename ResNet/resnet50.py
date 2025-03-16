import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, criterion, optimizer):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_val_acc = 0.0
        self.save_path = "C:\\Users\\Agah\\PycharmProjects\\Agah-StrokeClassification\\ViT\\runs\\best_model.pth"
        self.start_epoch = 0

        self.load_best_model()

    def train(self, num_epochs):

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        for epoch in range(self.start_epoch,num_epochs):
            print(f"Epoch {epoch+1} started")
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

            scheduler.step(val_loss)

            self.save_best_model(val_acc,epoch,train_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")

        print("Training is over")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        return running_loss / len(self.train_loader), train_accuracy

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        return val_loss / len(self.val_loader), val_accuracy

    def save_best_model(self, val_acc,epoch,loss):

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'loss': loss
            }, self.save_path)

            print("New best model saved")

    def load_best_model(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            print(f"Best model loaded. Proceeding from epoch {self.start_epoch}")
        else:
            print("No saved model found, training from scratch")


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder("C:\\Users\\Agah\\Desktop\\son_veriler\\train", transform=transform)
    test_dataset = datasets.ImageFolder("C:\\Users\\Agah\\Desktop\\son_veriler\\test", transform=transform)
    val_dataset = datasets.ImageFolder("C:\\Users\\Agah\\Desktop\\son_veriler\\validation", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    trainer = Trainer(model, train_loader, val_loader, device, criterion, optimizer)

    trainer.train(30)

