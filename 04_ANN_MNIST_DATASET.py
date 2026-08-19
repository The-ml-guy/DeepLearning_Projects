# ANN 
# 1. Download and preprocess a native PyTorch dataset (MNIST Handwritten Digits)
# We flatten the 28x28 image grids into a single 784-dimension vector line


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

#
# We flatten the 28x28 image grids into a single 784-dimension vector line
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)), # Standard MNIST normalization values
    transforms.Lambda(lambda x: torch.flatten(x)) 
])

# Load entire structural dataset
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 2. Native PyTorch Split (80% Train, 20% Validation/Test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 3. Create PyTorch DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 4. Define the ANN Architecture (Multi-class Classification for digits 0-9)
class PyTorchOnlyANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PyTorchOnlyANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Hidden Layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Raw logit output (No Softmax needed with CrossEntropyLoss)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize model (784 inputs -> 10 output digit categories)
model = PyTorchOnlyANN(input_dim=784, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 5. Native Training Loop
epochs = 5
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_images, batch_labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

# 6. Evaluation Loop using core torch matrices
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # Choose the highest logit score index as the prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy across MNIST: {accuracy:.2f}%")
