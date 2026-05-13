import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 第一部分：定义模型 (SimpleCNN)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第1层：卷积层。输入1个通道(灰度)，输出16个特征图，卷积核大小3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 激活函数
        self.relu = nn.ReLU()
        # 池化层（降采样，把图片变小）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第2层：全连接层（分类器）。经过池化后，28x28的图变成了14x14。
        self.fc = nn.Linear(16 * 14 * 14, 10) # 10代表10个类别(0-9)

    def forward(self, x):
        # x.shape: [Batch_size, 1, 28, 28]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x.shape: [Batch_size, 16, 14, 14]
        
        # 展平操作：把 2D 特征图拉平成 1D 向量
        x = x.view(x.size(0), -1)  
        # x.shape: [Batch_size, 3136]
        
        x = self.fc(x)
        # x.shape: [Batch_size, 10]
        return x

# ==========================================
# 第二部分：准备训练三大件
# ==========================================
model = SimpleCNN()
criterion = nn.CrossEntropyLoss() # 用于分类的交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.005) # Adam优化器，学习率0.005

# ==========================================
# 第三部分：准备数据 (我们模拟生成 MNIST 数据)
# ==========================================
print("正在生成模拟 MNIST 训练数据...")
# 模拟 100 张 28x28 的灰度图片
train_images = torch.randn(100, 1, 28, 28)
# 对应这100张图片的随机 0-9 标签
train_labels = torch.randint(0, 10, (100,)) 

# ==========================================
# 第四部分：标准训练循环 (Training Loop)
# ==========================================
epochs = 15 # 学 15 遍所有数据

print("开始训练 (CNN)！注意观察 Loss 变化：")
for epoch in range(epochs):
    # --- 动作 1: 前向传播 ---
    predictions = model(train_images)

    # --- 动作 2: 计算误差 ---
    loss = criterion(predictions, train_labels)

    # --- 动作 3: 清空梯度 ---
    optimizer.zero_grad()

    # --- 动作 4: 反向传播 (核心数学魔法！) ---
    loss.backward()

    # --- 动作 5: 更新参数 ---
    optimizer.step()

    # 打印当前的误差
    if (epoch + 1) % 1 == 0:
        print(f"第 {epoch+1:2d} 轮学习结束 | 当前误差 (Loss): {loss.item():.4f}")

print("CNN 模拟训练完成！")