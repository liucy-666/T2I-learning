import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 第一部分：定义模型 (SimpleTransformerBlock)
# ==========================================
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleTransformerBlock, self).__init__()
        # 1. 核心：多头自注意力层
        # batch_first=True 意思是输入数据的第一个维度是 Batch size
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # 2. 残差连接 + 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 3. 前馈神经网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # x.shape: [Batch_size, Seq_len, Embed_dim] (例如 [1, 20, 128])
        
        # --- 动作 A: 自注意力计算 ---
        # Q, K, V 都来自 x 本身
        attn_output, _ = self.attention(x, x, x)
        
        # --- 动作 B: 残差 + 归一化 1 ---
        x = self.norm1(x + attn_output) # 残差：x = x + attn_output
        
        # --- 动作 C: 前馈网络计算 ---
        ffn_output = self.ffn(x)
        
        # --- 动作 D: 残差 + 归一化 2 ---
        x = self.norm2(x + ffn_output) # 残差：x = x + ffn_output
        
        # x.shape 保持不变，但每个 Token 的向量都融合了全局的上下文信息
        return x

# ==========================================
# 第二部分：准备训练三大件
# ==========================================
embed_dim = 128  # 词向量维度
num_heads = 4    # 注意力头的数量
model = SimpleTransformerBlock(embed_dim, num_heads)

# 这里我们将任务简化为回归任务，预测一模一样的特征图
criterion = nn.MSELoss() # 均方误差损失函数，常用于回归
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 第三部分：准备数据 (我们模拟生成一个向量序列)
# ==========================================
print("正在生成模拟 Transformer 训练数据...")
# 模拟 1 个句子，长度为 20 个词，每个词已被转成了 128 维的向量
train_sequence = torch.randn(1, 20, embed_dim) 
# 我们让模型学着预测它自己 (回归任务)
train_label = train_sequence

# ==========================================
# 第四部分：标准训练循环
# ==========================================
epochs = 15

print("开始训练 (Transformer Block)！注意观察 Loss 变化：")
for epoch in range(epochs):
    # --- 动作 1: 前向传播 ---
    prediction = model(train_sequence)

    # --- 动作 2: 计算误差 ---
    loss = criterion(prediction, train_label)

    # --- 动作 3: 清空梯度 ---
    optimizer.zero_grad()

    # --- 动作 4: 反向传播 ---
    loss.backward()

    # --- 动作 5: 更新参数 ---
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"第 {epoch+1:2d} 轮学习结束 | 当前误差 (Loss): {loss.item():.4f}")

print("Transformer Block 模拟训练完成！")