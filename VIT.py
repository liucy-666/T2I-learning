import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ==========================================
# 第一部分：模型架构定义 (Vision Transformer)
# ==========================================

class PatchEmbedding(nn.Module):
    """图像切块与线性映射模块"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # 使用 Conv2d 实现不重叠的切块并映射维度
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # [B, 64, 4, 4]
        x = x.flatten(2)            # [B, 64, 16]
        x = x.transpose(1, 2)       # [B, 16, 64]
        return x

class MNIST_ViT(nn.Module):
    """专为 MNIST 定制的轻量级 ViT"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64, 
                 depth=6, num_heads=8, num_classes=10):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 董事长 CLS 令牌与 1D 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer 编码器 (Pre-Norm 架构)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, activation="gelu", 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        # 提取 CLS 令牌特征并分类
        cls_output = self.norm(x[:, 0])
        logits = self.head(cls_output)
        return logits

# ==========================================
# 第二部分：注意力特征提取与可视化工具
# ==========================================

def get_attention_map(model, image_tensor, device):
    """手动提取最后一层的 CLS 注意力权重"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 前置处理
        x = model.patch_embed(image_tensor)
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        
        # 跑完前面的 Transformer 层
        for i in range(len(model.transformer.layers) - 1):
            x = model.transformer.layers[i](x)
            
        # 在最后一层截取 Q 和 K 计算 Attention
        last_layer = model.transformer.layers[-1]
        x_norm = last_layer.norm1(x)
        
        qkv_weight = last_layer.self_attn.in_proj_weight
        qkv_bias = last_layer.self_attn.in_proj_bias
        qkv = F.linear(x_norm, qkv_weight, qkv_bias)
        
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = model.transformer.layers[0].self_attn.embed_dim // model.transformer.layers[0].self_attn.num_heads
        
        # 截取第一个头
        q_head0 = q[0, :, :head_dim]
        k_head0 = k[0, :, :head_dim]
        
        # Q * K^T / sqrt(d) -> Softmax
        attention_scores = torch.matmul(q_head0, k_head0.transpose(0, 1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 返回 CLS 对 16 个图块的注意力
        cls_attention = attention_weights[0, 1:]
        return cls_attention.cpu().numpy()

def visualize_overlay(original_image_tensor, attention_array):
    """绘制原图、热力图及叠加效果"""
    img_np = original_image_tensor.squeeze().cpu().numpy()
    attn_grid = attention_array.reshape(4, 4)
    attn_map_resized = cv2.resize(attn_grid, (28, 28), interpolation=cv2.INTER_CUBIC)
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attn_map_resized, cmap='jet')
    axes[1].set_title('Attention Map (4x4 to 28x28)')
    axes[1].axis('off')
    
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(attn_map_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 第三部分：主程序 (数据流水线、训练与测试)
# ==========================================

if __name__ == "__main__":
    # 1. 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用计算设备: {device}")
    
    # 2. 数据准备
    print("[*] 正在准备 MNIST 数据集...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    # 3. 模型实例化
    model = MNIST_ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练循环
    num_epochs = 5
    print("\n========== 开始训练 ==========")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
                
    # 5. 测试循环
    print("\n========== 开始在测试集上评估 ==========")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"[*] 最终测试集准确率: {100 * correct / total:.2f}%")
    
    # 6. 注意力热力图可视化 (抓取测试集的一张图)
    print("\n[*] 提取注意力热力图并进行可视化...")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    single_image = images[0:1] # 获取形状 [1, 1, 28, 28] 的单张张量
    
    attn_weights = get_attention_map(model, single_image, device)
    visualize_overlay(single_image, attn_weights)
    
    print("\n[*] 全部流程执行完毕！")