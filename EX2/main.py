'''
Author: foreverstyle
Date: 2025-05-08 08:11:48
LastEditTime: 2025-05-10 01:34:24
Description: 完成EX2 编写训练代码
FilePath: \ECE371\Assignment1\EX2\main.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

# Set data directory
data_dir = 'EX2/flower_dataset'

# Data augmentation and normalization for training and validation
data_transforms = transforms.Compose([
        # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
        ### START SOLUTION HERE ###
        # Add five data augmentation methods, Normalizating and Tranform to tensor
        transforms.RandomResizedCrop(224), # 随机尺寸裁剪到 224x224
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 颜色抖动
        transforms.RandomRotation(15), # 随机旋转 +/- 15 度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 随机仿射变换 (这里是随机平移)
        transforms.ToTensor(), #转换为Tensor，并缩放到[0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet的均值和标准差，用于归一化
        ### END SOLUTION HERE ###
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# Automatically split into 80% train and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Use DataLoader for both train and validation datasets
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Get class names from the dataset
class_names = full_dataset.classes

# Load pre-trained model and modify the last layer
model = models.resnet18(pretrained=True)


# GRADED FUNCTION: Modify the last fully connected layer of model
### START SOLUTION HERE ###
# Modify the last fully connected layer of model
num_ftrs = model.fc.in_features # 获取全连接层输入特征的数量
model.fc = nn.Linear(num_ftrs, len(class_names)) # 将全连接层替换为一个新的线性层，输出维度为类别数量

### END SOLUTION HERE ###



# GRADED FUNCTION: Define the loss function
### START SOLUTION HERE ###
# Define the loss function
criterion = nn.CrossEntropyLoss() #交叉熵损失函数

### END SOLUTION HERE ###

# GRADED FUNCTION: Define the optimizer
### START SOLUTION HERE ###
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # 使用SGD优化器

### END SOLUTION HERE ###

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Print learning rate for current epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        
                        # GRADED FUNCTION: Backward pass and optimization
                        ### START SOLUTION HERE ###
                        # Backward pass and optimization
                        loss.backward() # 计算梯度
                        optimizer.step() # 更新参数

                        ### END SOLUTION HERE ###

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # Update learning rate based on scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if validation accuracy is the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)

               # GRADED FUNCTION: Save the best model
                ### START SOLUTION HERE ###
                # Save the best model
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth')) # 将模型保存为 best_model.pth 文件
                ### END SOLUTION HERE ###

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__': #windows下使用多线程需要
    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)