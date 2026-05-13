import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. 我们的“玩具数据集” (高度规律化)
# ==========================================
text = """
春天来了，小草绿了，花儿开了。
夏天来了，天气热了，知了叫了。
秋天来了，树叶黄了，果子熟了。
冬天来了，下大雪了，天气冷了。
春天来了，天气暖了，燕子飞了。
夏天来了，太阳烈了，西瓜甜了。
秋天来了，稻子黄了，农民笑了。
冬天来了，北风吹了，梅花开了。
"""
# 去除多余的换行符，让文本连贯
text = text.replace('\n', '').strip()
print(f"语料库加载完毕，共 {len(text)} 个字符。")

chars = tuple(set(text))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
text_as_int = [char2idx[c] for c in text]

# ==========================================
# 2. 定义 RNN 模型 (和刚才完全一样)
# ==========================================
class TextGeneratorRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TextGeneratorRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden) 
        out = self.fc(out) 
        return out, hidden

# ==========================================
# 3. 设置训练参数 (专为小文本优化)
# ==========================================
seq_length = 4    # 句子很短，每次看 4 个字就够了
embed_dim = 16    # 词汇量少，维度不需要太大
hidden_dim = 64
epochs = 1500     # 跑 1500 轮，因为数据量小，几秒钟就能跑完

model = TextGeneratorRNN(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # 学习率调大一点，学得快

loss_history = []

# ==========================================
# 4. 闪电训练！
# ==========================================
print("\n开始闪电训练...")
for epoch in range(epochs):
    start_idx = random.randint(0, len(text) - seq_length - 1)
    chunk_in = text_as_int[start_idx : start_idx + seq_length]
    chunk_out = text_as_int[start_idx + 1 : start_idx + seq_length + 1]

    x = torch.tensor(chunk_in).unsqueeze(0)
    y = torch.tensor(chunk_out).unsqueeze(0)

    predictions, _ = model(x)
    loss = criterion(predictions.view(-1, vocab_size), y.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

    if (epoch + 1) % 300 == 0:
        print(f"第 {epoch+1:4d} 轮 | 当前误差 (Loss): {loss.item():.4f}")

# ==========================================
# 5. 见证模型“写诗”
# ==========================================
print("\n--- 训练结束，开始自动生成文本 ---")

def generate_text(model, start_string, generate_length=30):
    model.eval()
    input_eval = [char2idx[s] for s in start_string]
    input_eval = torch.tensor(input_eval).unsqueeze(0)
    generated_text = start_string
    hidden = None 

    for i in range(generate_length):
        with torch.no_grad():
            predictions, hidden = model(input_eval, hidden)
        
        next_token_logits = predictions[0, -1, :]
        predicted_id = torch.argmax(next_token_logits).item()
        
        generated_text += idx2char[predicted_id]
        input_eval = torch.tensor([[predicted_id]])
        
    return generated_text

# 给它一个起手式，让它自己往下接
seed = "夏天"
print(f"给定开头：'{seed}'")
print(f"RNN 生成结果：\n{generate_text(model, start_string=seed, generate_length=40)}")

plt.figure(figsize=(6, 3))
plt.plot(loss_history, color='orange')
plt.title("Toy Dataset: RNN Text Generation Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()