import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets,transforms
import os


if __name__ == '__main__':
    model = models.densenet121(weights=True)

    model.features.conv0 = nn.Conv2d(in_channels=1,
                                     out_channels=64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3,
                                     bias=False)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs,1)

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
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)


    def train(num_epochs):

        if os.path.exists("model_checkpoint.pth"):
            checkpoint = torch.load("model_checkpoint.pth", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Proceeding from checkpoint")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            print(f"Starting Epoch {epoch + 1}")

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
            val()

        torch.save(model.state_dict(), "model_weights.pth")
        print("Finished Training")

    def val():
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy:.4f}")


    print("Finished Training")

    def test():
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_accuracy = correct / total
        print(f"Test Accuracy: {test_accuracy:.4f}")


    num_epochs = 30

    train(num_epochs)