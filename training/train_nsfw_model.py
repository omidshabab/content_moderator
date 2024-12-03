import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class NSFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir should have two subdirectories:
        - safe/: containing safe images
        - unsafe/: containing NSFW images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['safe', 'unsafe']
        
        self.file_list = []
        self.labels = []
        
        # Load safe images
        safe_dir = os.path.join(root_dir, 'safe')
        for filename in os.listdir(safe_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.file_list.append(os.path.join(safe_dir, filename))
                self.labels.append(0)  # 0 for safe
        
        # Load unsafe images
        unsafe_dir = os.path.join(root_dir, 'unsafe')
        for filename in os.listdir(unsafe_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.file_list.append(os.path.join(unsafe_dir, filename))
                self.labels.append(1)  # 1 for unsafe

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ModelTrainer:
    def __init__(self, train_dir, valid_dir, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create datasets
        self.train_dataset = NSFWDataset(train_dir, self.transform)
        self.valid_dataset = NSFWDataset(valid_dir, self.transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Initialize model
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Modify classifier for binary classification
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(self.model.last_channel, 2)
        )
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=2, factor=0.5
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        return running_loss/len(self.train_loader), 100.*correct/total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader, desc='Validating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss/len(self.valid_loader), 100.*correct/total

    def train(self, num_epochs=10, save_dir='models/ml'):
        best_valid_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.validate()
            
            print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
            print(f'Valid Loss: {valid_loss:.4f} Valid Acc: {valid_acc:.2f}%')
            
            # Learning rate scheduling
            self.scheduler.step(valid_loss)
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.model.state_dict(), 
                         os.path.join(save_dir, 'nsfw_mobilenet.pth'))
                print('Model saved!')

def main():
    # Set paths to your dataset directories
    train_dir = 'training/dataset/train'  # Should contain 'safe' and 'unsafe' subdirectories
    valid_dir = 'training/dataset/valid'  # Should contain 'safe' and 'unsafe' subdirectories
    
    # Initialize trainer
    trainer = ModelTrainer(train_dir, valid_dir)
    
    # Start training
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main()