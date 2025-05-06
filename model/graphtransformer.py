import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 1. 单层 Graph Transformer
# ---------------------------------------------------------------------
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_ff=2048, dropout=0.1):
        """
        Args:
            dim      : 输入/输出的特征维度 d
            heads    : 注意力头数 h
            dim_ff   : 前馈网络隐藏层宽度
            dropout  : dropout概率
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # QKV 投影
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # 前馈
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, adj):
        """
        Args:
            x   : 节点特征 (N, d)
            adj : 邻接(掩码)矩阵 (N, N)；0-1 或 bool。应含自环。
        Returns:
            out : (N, d)
        """
        # ---- 多头注意力 ----
        h = self.heads
        N, _ = x.shape
        qkv = self.to_qkv(self.norm1(x))           # (N, 3d)
        qkv = qkv.reshape(N, 3, h, self.dim // h)  # (N, 3, h, d_h)
        q, k, v = qkv.unbind(dim=1)                # (N, h, d_h)

        # 缩放点积注意力
        attn = torch.einsum("nhd,mhd->hnm", q, k) * self.scale   # (h, N, N)

        # 邻接掩码：不连通处置为 -inf
        mask = (~adj.bool()).unsqueeze(0)          # (1, N, N)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 聚合
        out = torch.einsum("hnm,mhd->nhd", attn, v)  # (N, h, d_h)
        out = out.reshape(N, self.dim)
        out = self.proj_drop(self.proj(out))

        # 残差 + 前馈
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x

# ---------------------------------------------------------------------
# 2. 堆叠成完整 Graph Transformer 网络
# ---------------------------------------------------------------------
class GraphTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        dim_ff,
        num_classes,
        dropout=0.1,
        readout="mean",   # "mean" | "sum" | "max"
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(dim, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.readout = readout
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, adj, mask=None):
        """
        Args:
            x    : (N, d) 节点特征
            adj  : (N, N) 邻接掩码（需含对角线）
            mask : (N,) 可选，图中有效节点的 bool 掩码（处理 padding 图）
        """
        for layer in self.layers:
            x = layer(x, adj)

        # -------- 图级汇聚 ----------
        if mask is not None:
            x_pool = x[mask]
        else:
            x_pool = x

        if self.readout == "mean":
            g = x_pool.mean(dim=0)
        elif self.readout == "sum":
            g = x_pool.sum(dim=0)
        elif self.readout == "max":
            g, _ = x_pool.max(dim=0)
        else:
            raise ValueError("Unknown readout type")

        return self.classifier(g)

# ---------------------------------------------------------------------
# 3. 简单测试
# ---------------------------------------------------------------------
if __name__ == "__main__":
    N = 16            # 节点数
    dim = 64          # 特征维
    num_classes = 3

    # 随机图
    adj = torch.rand(N, N) < 0.3
    adj |= torch.eye(N, dtype=torch.bool)         # 一定要包含自环

    # 随机节点特征
    x = torch.randn(N, dim)

    model = GraphTransformer(
        num_layers=4,
        dim=dim,
        num_heads=8,
        dim_ff=256,
        num_classes=num_classes,
        dropout=0.1,
        readout="mean",
    )

    logits = model(x, adj)
    print("logits:", logits.shape)     # (num_classes,)
