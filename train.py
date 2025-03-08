import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet
from dataset import RetinalDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 16
epochs = 50
learning_rate = 1e-4
train_data_dir = './train_data'
val_data_dir = './val_data'
model_save_path = './unet_model.pth'

# Transformations and augmentations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare datasets and dataloaders
train_dataset = RetinalDataset(train_data_dir, transform=transform)
val_dataset = RetinalDataset(val_data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize U-Net model
model = UNet(in_channels=3, out_channels=3, base_filters=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Change if binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_one_epoch(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    return running_loss / len(train_loader)

# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

# Plot loss curve
def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# Training and evaluation
train_losses = []
val_losses = []
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Save the model every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch}")
    
# Plot the loss curves
plot_loss(train_losses, val_losses)

