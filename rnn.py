import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 第一部分：定义模型 (SimpleRNN)
# ==========================================
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(SimpleRNN, self).__init__()
        # 1. 词嵌入层：把词ID变成向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 2. RNN核心层：处理序列。batch_first=True 表示输入形状为 (Batch, Seq, Feature)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        # 3. 分类器：基于全句记忆给出分类
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x.shape: [Batch_size, Seq_len] (例如 [1, 20])
        embedded = self.embedding(x)
        # embedded.shape: [Batch_size, Seq_len, Embed_dim]
        
        # out 包含所有时刻的状态，hidden 是最后一个时刻的记忆
        out, hidden = self.rnn(embedded)
        # out.shape: [Batch_size, Seq_len, Hidden_dim]
        # hidden.shape: [1, Batch_size, Hidden_dim] (假设单层RNN)
        
        # 我们用最后一个时刻的记忆作为整句话的特征，输入分类器
        final_state = hidden.squeeze(0) # [Batch_size, Hidden_dim]
        result = self.fc(final_state)
        # result.shape: [Batch_size, Num_classes]
        return result

# ==========================================
# 第二部分：准备训练三大件
# ==========================================
vocab_size = 1000 # 词汇表大小
embed_dim = 32 # 词向量维度
hidden_dim = 64 # RNN隐藏层维度
num_classes = 2 # 情感类别数 (正向/负向)

model = SimpleRNN(vocab_size, embed_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 第三部分：准备数据 (我们模拟生成情感数据)
# ==========================================
print("正在生成模拟情感分析训练数据...")
# 模拟 1 个句子，长度为 20 个词，里面填满了 0-999 随机的词 ID
train_sentence = torch.randint(0, vocab_size, (1, 20)) 
# 模拟这个句子的标签 (例如，0代表负向)
train_label = torch.tensor([0]) 

# ==========================================
# 第四部分：标准训练循环
# ==========================================
epochs = 15

print("开始训练 (RNN)！注意观察 Loss 变化：")
for epoch in range(epochs):
    # --- 动作 1: 前向传播 ---
    prediction = model(train_sentence)

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

print("RNN 模拟训练完成！")