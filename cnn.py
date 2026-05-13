import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义模型 (SimpleCNN 保持原汁原味)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

# ==========================================
# 2. 下载并加载【真实】 MNIST 数据集
# ==========================================
print("正在连接网络下载真实的 MNIST 数据集 (约 60MB)...")

# transform 将图片转换为模型能懂的 Tensor，并把像素值缩放到 0~1 之间
transform = transforms.Compose([transforms.ToTensor()])

# 自动下载训练集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# DataLoader 就像一个发牌员，把 60000 张图分成每批 64 张发给模型
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ==========================================
# 3. 准备三大件并开始训练
# ==========================================
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

loss_history = [] 
epochs = 2 # 真实数据有 60000 张，跑 2 遍所有数据就足够看到效果了

print("开始真实数据训练！这次的 Loss 下降会非常平稳：")
for epoch in range(epochs):
    # 遍历发牌员发出的每一批图片 (images 包含 64 张图，labels 包含 64 个答案)
    for batch_idx, (images, labels) in enumerate(train_loader):
        
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每隔 100 批次记录一次 Loss 并打印
        if batch_idx % 100 == 0:
            loss_history.append(loss.item())
            print(f"第 {epoch+1} 轮 | 进度: {batch_idx * 64}/60000 | 当前误差: {loss.item():.4f}")

print("真实数据训练完成！马上开始画图...")

# ==========================================
# 4. 完美无重叠的可视化排版
# ==========================================
# 创建一个 1 行 5 列的画布
fig, axes = plt.subplots(1, 5, figsize=(16, 3), gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})

# --- 第一块：画 Loss 曲线 ---
axes[0].plot(loss_history, color='blue', linewidth=2)
axes[0].set_title("Real MNIST Training Loss")
axes[0].set_xlabel("Steps (x100)")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# --- 后面四块：画卷积核 ---
kernels = model.conv1.weight.detach().numpy()
for i in range(4):
    kernel_image = kernels[i][0] 
    axes[i+1].imshow(kernel_image, cmap='gray')
    axes[i+1].set_title(f"Kernel {i+1}")
    axes[i+1].axis('off')

# 自动调整间距，绝对不重叠
plt.tight_layout()
plt.show()