# verify_model.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import random_split # Still need random_split to get val_dataset size if not using separate file
import os

# --- Configuration ---

# <--- !!! 请修改为您的花卉数据集在运行此脚本时的正确路径 !!! --->
# 例如，如果您的 verify_model.py 也在 EX2 文件夹里，且数据集在 EX2/flower_dataset
data_dir = './flower_dataset'

# <--- !!! 请修改为你要验证的 .pth 文件路径 !!! --->
# 例如，验证 Exercise 2 训练保存的最佳模型:
model_to_verify_path = './work_dir/best_model.pth'
# 例如，验证 Exercise 1 训练保存的最佳模型 (如果放在某个路径下):
# model_to_verify_path = 'path/to/your/ex1_best_model.pth'


batch_size = 32 # 验证时的批量大小，可以根据需要调整

# --- Dataset Loading (Validation Set Only) ---

# Data transformations for validation (similar to main.py)
# 不需要训练时的随机增强，只需要标准的缩放和中心裁剪
val_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), # 虽然是验证，但使用 ImageFolder 需要 transforms
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 注意：这里为了获取数据集的类别名称和大小，我们仍然加载整个数据集并进行划分
# 如果你已经有了单独的 val 数据集文件夹，并且不需要动态划分，可以简化这部分代码
full_dataset = datasets.ImageFolder(data_dir, val_transforms)

# 动态划分 (与 main.py 保持一致以便获取相同的验证集大小和类别名称)
# 如果你的 val 数据集是固定的文件，请调整这部分代码
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
# 注意：这里 random_split 会每次生成不同的划分。
# 如果你需要一个固定的验证集进行评估，应该提前保存好验证集的文件列表或使用固定的随机种子。
# 为了简单演示，我们沿用 main.py 的划分方式。
_, val_dataset = random_split(full_dataset, [train_size, val_size])


val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False, # 验证时不打乱
    num_workers=0, # 验证时通常 num_workers=0 即可，避免多进程问题
    # pin_memory=True if torch.cuda.is_available() else False,
)

# Get class names and dataset size from the dataset
class_names = full_dataset.classes
val_dataset_size = len(val_dataset)


# --- Model Definition ---

# <--- !!! 请根据你要验证的 .pth 文件对应的模型来定义这里的模型结构 !!! --->
# 如果 .pth 文件是 Exercise 2 的 ResNet18 (通过 main.py 训练得到):
model = models.resnet18(pretrained=False) # 只需要结构，不需要预训练权重
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names)) # 修改头部适应你的类别数

# 如果 .pth 文件是 Exercise 1 的 ResNet50 (通过 MMClassification 训练得到):
# model = models.resnet50(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(class_names)) # 修改头部适应你的类别数


# --- Evaluation Function ---

def evaluate_model(model, dataloader, dataset_size):
    """Evaluates the model on the given dataloader."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluate mode

    running_corrects = 0
    total_samples = 0

    print("Evaluating model on validation set...")
    # 在评估阶段不需要计算梯度，可以加快速度和节省内存
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # 获取预测的类别

            running_corrects += torch.sum(preds == labels.data) # 累加预测正确的数量
            total_samples += inputs.size(0) # 累加总样本数

    accuracy = running_corrects.double() / total_samples # 计算准确率
    print(f"Validation Accuracy: {accuracy:.4f}")

    return accuracy.item() # 返回准确率的数值


# --- Main Execution ---

if __name__ == '__main__':
    # 检查模型文件是否存在
    if not os.path.exists(model_to_verify_path):
        print(f"Error: Model file not found at {model_to_verify_path}")
    else:
        # 加载模型权重
        try:
            # torch.load() 会加载整个 state_dict
            model.load_state_dict(torch.load(model_to_verify_path))
            print(f"Successfully loaded model state_dict from {model_to_verify_path}")

            # 执行评估
            accuracy = evaluate_model(model, val_loader, val_dataset_size)

            if accuracy is not None:
                 print(f"\nVerification complete. Model at {model_to_verify_path} has Validation Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
            print("Please ensure the model architecture definition in verify_model.py")
            print(f"matches the model that saved the state_dict in {model_to_verify_path}")
