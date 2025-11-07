"""
Plant Disease Classification using ResNet-9
===========================================

This script trains a ResNet-9 model for plant disease classification.

Requirements:
- torch
- torchvision
- numpy
- pandas
- matplotlib
- Pillow
- torchsummary (install with: pip install torchsummary)

Dataset Structure:
- Place your dataset in a folder with the following structure:
  dataset/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── valid/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        └── test/
            ├── image1.jpg
            ├── image2.jpg
            └── ...

Usage:
    python plant_disease_classification.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchsummary import summary


# ============================================================================
# CONFIGURATION
# ============================================================================

# Update these paths according to your dataset location
DATA_DIR = "../New Plant Diseases Dataset"  # Updated to point to actual dataset
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Training parameters
RANDOM_SEED = 7
BATCH_SIZE = 32
EPOCHS = 10
MAX_LR = 0.001  # Reduzido de 0.01 para evitar explosão de gradientes
GRAD_CLIP = 0.1
WEIGHT_DECAY = 5e-4  # Aumentado de 1e-4 para mais regularização
INPUT_SHAPE = (3, 256, 256)

# Model save path
MODEL_PATH = './plant-disease-model.pth'
MODEL_COMPLETE_PATH = './plant-disease-model-complete.pth'


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def accuracy(outputs, labels):
    """Calculate accuracy"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    """Base class for image classification models"""
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


def ConvBlock(in_channels, out_channels, pool=False):
    """Convolution block with BatchNormalization"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    """ResNet-9 architecture for image classification"""
    
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim: 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim: 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim: 512 x 4 x 4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        """Forward pass"""
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader):
    """Evaluate the model on validation set"""
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
    """Train the model using One Cycle learning rate policy"""
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Scheduler for one cycle learning rate
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                 steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # Recording and updating learning rates
            lrs.append(get_lr(optimizer))
            sched.step()
            
        # Validation
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def denormalize(tensor):
    """Desnormaliza tensor para visualização"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def show_image(image, label, classes):
    """Display a single image with its label"""
    print("Label: " + classes[label] + " (" + str(label) + ")")
    plt.imshow(denormalize(image).permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def show_batch(data):
    """Show a batch of training instances"""
    for images, labels in data:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xticks([])
        ax.set_yticks([])
        # Desnormaliza as imagens antes de mostrar
        denorm_images = torch.stack([denormalize(img) for img in images[:32]])
        ax.imshow(make_grid(denorm_images, nrow=8).permute(1, 2, 0))
        plt.show()
        break


def plot_accuracies(history):
    """Plot accuracy vs epochs"""
    accuracies = [x['val_accuracy'] for x in history]
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. No. of Epochs')
    plt.grid(True)
    plt.show()


def plot_losses(history):
    """Plot training and validation losses"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, '-bx', label='Training')
    plt.plot(val_losses, '-rx', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. No. of Epochs')
    plt.grid(True)
    plt.show()


def plot_lrs(history):
    """Plot learning rate schedule"""
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.xlabel('Batch No.')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Batch No.')
    plt.grid(True)
    plt.show()


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_image(img, model, classes, device):
    """Predict the class of a single image"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("=" * 80)
    print("Plant Disease Classification using ResNet-9")
    print("=" * 80)
    
    # Check if dataset exists
    if not os.path.exists(TRAIN_DIR):
        print(f"\nError: Training directory not found at {TRAIN_DIR}")
        print("Please update DATA_DIR in the script to point to your dataset.")
        return
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    
    # Get device
    device = get_default_device()
    print(f"\nUsing device: {device}")
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    print("\nLoading dataset...")
    
    # Transformações com normalização ImageNet para estabilidade
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    valid_dataset = ImageFolder(VALID_DIR, transform=valid_transform)

    diseases = os.listdir(TRAIN_DIR)
    print(f"Total disease classes: {len(diseases)}")
    
    # Get statistics
    plants = []
    num_diseases = 0
    for plant in diseases:
        if plant.split('___')[0] not in plants:
            plants.append(plant.split('___')[0])
        if plant.split('___')[1] != 'healthy':
            num_diseases += 1
    
    print(f"Number of unique plants: {len(plants)}")
    print(f"Number of diseases: {num_diseases}")
    
    # Count images per class
    nums = {}
    for disease in diseases:
        nums[disease] = len(os.listdir(os.path.join(TRAIN_DIR, disease)))
    
    n_train = sum(nums.values())
    print(f"Total training images: {n_train}")
    
    # ========================================================================
    # CREATE DATA LOADERS
    # ========================================================================
    print("\nCreating data loaders...")
    
    # Only use pin_memory with CUDA, not with MPS
    use_pin_memory = device.type == 'cuda'

    train_dl = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=use_pin_memory)
    valid_dl = DataLoader(valid_dataset, BATCH_SIZE,
                         num_workers=2, pin_memory=use_pin_memory)

    # Move to device
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("\nCreating model...")
    
    model = to_device(ResNet9(3, len(train_dataset.classes)), device)
    print(model)
    
    # Print model summary
    if torch.cuda.is_available():
        print("\nModel Summary:")
        print(summary(model.cuda(), INPUT_SHAPE))
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 80)
    
    history = fit_OneCycle(
        EPOCHS, 
        MAX_LR, 
        model, 
        train_dl, 
        valid_dl,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        opt_func=torch.optim.Adam
    )
    
    print("-" * 80)
    print("Training completed!")
    
    # ========================================================================
    # VISUALIZE RESULTS
    # ========================================================================
    print("\nGenerating training plots...")
    
    plot_accuracies(history)
    plot_losses(history)
    plot_lrs(history)
    
    # ========================================================================
    # TEST MODEL
    # ========================================================================
    if os.path.exists(TEST_DIR):
        print("\nTesting model on test set...")
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        test_dataset = ImageFolder(TEST_DIR, transform=test_transform)
        test_images = sorted(os.listdir(os.path.join(TEST_DIR, 'test')))
        
        # Predict first image
        img, label = test_dataset[0]
        plt.figure(figsize=(8, 8))
        plt.imshow(denormalize(img).permute(1, 2, 0))
        plt.axis('off')
        predicted = predict_image(img, model, train_dataset.classes, device)
        plt.title(f'Predicted: {predicted}')
        plt.show()
        
        # Predict all test images
        print("\nPredictions for test set:")
        for i, (img, label) in enumerate(test_dataset):
            predicted = predict_image(img, model, train_dataset.classes, device)
            print(f'Image: {test_images[i]}, Predicted: {predicted}')
    else:
        print(f"\nTest directory not found at {TEST_DIR}. Skipping testing.")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\nSaving model...")
    
    # Save model state dict
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model state dict saved to: {MODEL_PATH}")
    
    # Save complete model
    torch.save(model, MODEL_COMPLETE_PATH)
    print(f"Complete model saved to: {MODEL_COMPLETE_PATH}")
    
    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()
