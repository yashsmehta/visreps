import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


num_classes = 200
batch_size = 128
num_epochs = 20
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.272, 0.265, 0.274]) # these are the mean and std of the tiny-ImageNet dataset
])

train_dataset = datasets.ImageFolder(root='../data/tiny-imagenet-200/train', transform=transform)
test_dataset = datasets.ImageFolder(root='../data/tiny-imagenet-200/val', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = models.alexnet()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
print(model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

if device == 'cuda':
    print("Using GPU for training")
else:
    print("Using CPU for training")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

    # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
    # print(f'Model saved as model_epoch_{epoch+1}.pth')
