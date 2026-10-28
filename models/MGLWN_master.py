import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn import BatchNorm, global_mean_pool



class MGLWN_master(torch.nn.Module):
    def __init__(self, feature, out_channel, hop=3, fusion_type='moe'):
    # fusion_type 参数,可选：'moe', 'sum', 'mean', 'max', 'weighted_sum'
        super(MGLWN_master, self).__init__()

        self.GConv1 = LiftingLayer(feature, 1024, hop=hop, fusion_type=fusion_type)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = LiftingLayer(1024, 1024, hop=hop, fusion_type=fusion_type)
        self.bn2 = BatchNorm(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x, U, P = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x1 = global_mean_pool(x, batch)

        x, U, P = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x2 = global_mean_pool(x, batch)

        x = x1 + x2

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


class LiftingLayer(nn.Module):
    """
    fusion_type 可选值:
    - 'moe': 动态混合专家，每个节点有独立的 gate
    - 'sum': 直接求和
    - 'mean': 直接求平均
    - 'max': 最大池化
    - 'weighted_sum': 静态可学习权重的加权求和 (所有节点共享同一组权重)
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 hop=2,
                 alpha=None,
                 variant=False,
                 alpha_att=0.2,
                 fusion_type='moe'):
        super().__init__()

        self.variant = variant
        self.in_channels = in_channels * 2 if variant else in_channels
        self.out_channels = out_channels if out_channels is not None else self.in_channels
        self.hop = int(hop)
        self.alpha_ = alpha
        self.leaky_slope = alpha_att
        self.fusion_type = fusion_type

        # linear projection
        self.lin = nn.Linear(self.in_channels, self.in_channels, bias=False)

        # attention parameters
        self.att_left = nn.Parameter(torch.Tensor(self.in_channels, 1))
        self.att_right = nn.Parameter(torch.Tensor(self.in_channels, 1))

        # blending coefficients
        self.temp = nn.Parameter(torch.zeros(2))

        # chebyshev-like scaling
        init_cheb = 0.9 * (1 - 0.9) ** torch.arange(self.hop, dtype=torch.float32)
        self.cheb = nn.Parameter(init_cheb.clone())

        # ===== 融合策略层初始化 =====
        if self.fusion_type == 'moe':
            # 原始 MoE 门控: 输入 x -> 输出 hop 维权重
            self.moe_gate = nn.Sequential(
                nn.Linear(self.in_channels, self.hop),
                nn.Softmax(dim=-1)
            )
        elif self.fusion_type == 'weighted_sum':
            # 静态加权: 一个可学习的参数，维度为 [1, hop]，所有节点共享
            self.static_weight = nn.Parameter(torch.Tensor(1, self.hop))
            nn.init.xavier_uniform_(self.static_weight)


        # output projection
        self.out_lin = nn.Linear(self.in_channels, self.out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_left)
        nn.init.xavier_uniform_(self.att_right)
        nn.init.xavier_uniform_(self.out_lin.weight)
        if self.out_lin.bias is not None:
            nn.init.zeros_(self.out_lin.bias)
        self.temp.data.fill_(0.0)

        # 针对 weighted_sum 重置参数
        if self.fusion_type == 'weighted_sum':
            nn.init.constant_(self.static_weight, 1.0 / self.hop)  # 初始化为平均值


    def compute_edge_attention(self, x, edge_index):
        h = self.lin(x)
        left = (h @ self.att_left).squeeze(-1)
        right = (h @ self.att_right).squeeze(-1)
        src, dst = edge_index
        e = left[src] + right[dst]
        e = F.leaky_relu(e, negative_slope=self.leaky_slope)
        U = softmax(e, dst)
        P = 0.5 * U
        return U, P

    def message_aggregate(self, x, edge_index, edge_weight):
        src, dst = edge_index
        msg = x[src] * edge_weight.unsqueeze(-1)
        out = scatter_add(msg, dst, dim=0, dim_size=x.size(0))
        return out

    # ------------------------------------------------------------
    # Lifting + Fusion
    # ------------------------------------------------------------
    def forward_lifting_bases(self, x, edge_index, U, P):
        N, C = x.size()
        src, dst = edge_index

        coe = torch.sigmoid(self.temp)
        cheb = torch.sigmoid(self.cheb)
        rowsum = scatter_add(P, src, dim=0, dim_size=N)

        update = x
        expert_feats = []
        decom_feature = []

        # --- 1. 生成多跳专家特征---
        for k in range(self.hop):
            update = self.message_aggregate(update, edge_index, U)

            if self.alpha_ is None:
                feat_even = coe[0] * x + update
            else:
                feat_even = update

            rs = cheb[k - 1] * rowsum if k >= 1 else rowsum
            feat_odd = update - feat_even * rs.unsqueeze(-1)

            if self.alpha_ is None:
                feat_fuse = coe[1] * feat_even + (1.0 - coe[1]) * feat_odd
            else:
                feat_fuse = self.alpha_ * feat_even + (1.0 - self.alpha_) * feat_odd

            decom_feature.append([feat_even, feat_odd])
            expert_feats.append(feat_fuse)

        # --- 2. 融合---

        # experts 维度: [N, hop, C]
        experts = torch.stack(expert_feats, dim=1)
        gate = None  # 默认 gate 为空，除非是 MoE 或 weighted_sum

        if self.fusion_type == 'moe':
            # 原版: 每个节点根据特征计算权重 [N, hop]
            gate = self.moe_gate(x)
            # 加权求和
            feat_prime = torch.sum(experts * gate.unsqueeze(-1), dim=1)

        elif self.fusion_type == 'sum':
            # 变体1: 直接求和 [N, C]
            feat_prime = torch.sum(experts, dim=1)
            # 为了兼容接口返回一个假的 uniform gate 用于可视化
            gate = torch.ones(N, self.hop, device=x.device) / self.hop

        elif self.fusion_type == 'mean':
            # 变体2: 直接求平均 [N, C]
            feat_prime = torch.mean(experts, dim=1)
            gate = torch.ones(N, self.hop, device=x.device) / self.hop

        elif self.fusion_type == 'max':
            # 变体3: 最大池化 [N, C]
            # max 返回 (values, indices)，我们只需要 values
            feat_prime, _ = torch.max(experts, dim=1)
            gate = torch.zeros(N, self.hop, device=x.device)  # max 没有明确的 gate 概念

        elif self.fusion_type == 'weighted_sum':
            # 变体4: 静态加权求和
            # gate: [1, hop]，经过 Softmax 保证和为 1
            gate_weights = F.softmax(self.static_weight, dim=-1)
            # 扩展 gate 到 [N, hop] 方便计算，但每一行都是一样的
            gate = gate_weights.expand(N, -1)
            feat_prime = torch.sum(experts * gate.unsqueeze(-1), dim=1)

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return feat_prime, decom_feature, expert_feats, gate

    def forward(self, x, edge_index):
        U, P = self.compute_edge_attention(x, edge_index)
        feat, decom_feature, expert_feats, gate = self.forward_lifting_bases(x, edge_index, U, P)
        out = self.out_lin(feat)
        return out, U, P
