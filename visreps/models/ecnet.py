"""
ECTiedNet: Weight-tied Expansion-Contraction CNN
Minimal architecture with weight reuse, divisive normalization, and BlurPool downsampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn_groups_for(channels: int, max_groups: int = 16) -> int:
    """Largest divisor of `channels` not exceeding max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class DivisiveNorm(nn.Module):
    """Local gain control: y = x / (eps + avg_pool(|x|, k=3))."""
    def __init__(self, eps: float = 1e-3, kernel_size: int = 3):
        super().__init__()
        self.eps = eps
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (self.pool(x.abs()) + self.eps)


class BlurPool2d(nn.Module):
    """Anti-aliased downsampling (stride=2) with a fixed low-pass kernel.
    Uses a separable binomial [1, 2, 1] ⊗ [1, 2, 1] normalized kernel.
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        assert stride in (2, 3), "BlurPool2d currently supports stride 2 or 3"
        self.stride = stride
        # 1D binomial kernel [1, 2, 1]
        k1d = torch.tensor([1., 2., 1.])
        k2d = torch.einsum('i,j->ij', k1d, k1d)
        k2d = k2d / k2d.sum()
        self.register_buffer('kernel', k2d[None, None, :, :].repeat(channels, 1, 1, 1))
        self.groups = channels
        self.pad = (k2d.shape[-1] // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=self.pad, groups=self.groups)
        return x


class ECBlock(nn.Module):
    """
    1×1 expand (C→rC) → DW 3×3 (dilated) → 1×1 contract (rC→C),
    GN + SiLU, DivisiveNorm after DW, residual with layer-scale gamma.
    SAME weights are reused each time this block is called.
    """
    def __init__(self, channels: int, expansion: int = 6,
                 max_gn_groups: int = 16, layer_scale_init: float = 1e-3):
        super().__init__()
        C = channels
        Cexp = C * expansion

        # 1x1 expand (explicitly ungrouped)
        self.conv_expand = nn.Conv2d(C, Cexp, kernel_size=1, groups=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups_for(Cexp, max_gn_groups), Cexp)
        self.act = nn.SiLU(inplace=True)

        # depthwise 3x3 (weights reused with runtime dilation)
        self.dw_weight = nn.Parameter(torch.empty(Cexp, 1, 3, 3))
        self.dw_bias = nn.Parameter(torch.zeros(Cexp))
        nn.init.kaiming_normal_(self.dw_weight, mode='fan_out', nonlinearity='relu')

        # divisive normalization
        self.divnorm = DivisiveNorm(eps=1e-3, kernel_size=3)

        # 1x1 contract (explicitly ungrouped)
        self.conv_contract = nn.Conv2d(Cexp, C, kernel_size=1, groups=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups_for(C, max_gn_groups), C)

        # residual layer-scale
        self.gamma = nn.Parameter(torch.ones(1) * layer_scale_init)

        # init 1x1s
        nn.init.kaiming_normal_(self.conv_expand.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_contract.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        identity = x
        out = self.conv_expand(x)
        out = self.gn1(out)
        out = self.act(out)

        out = F.conv2d(out, self.dw_weight, bias=self.dw_bias,
                       stride=1, padding=dilation, dilation=dilation,
                       groups=out.shape[1])  # depthwise
        out = self.divnorm(out)

        out = self.conv_contract(out)
        out = self.gn2(out)

        return identity + self.gamma * out


class ECTiedNet(nn.Module):
    """
    Apply the SAME ECBlock N times with a dilation schedule and one mid BlurPool.
    Then GAP → 2-layer MLP (4096x4096) with dropout → classifier.
    """
    def __init__(self, num_classes: int = 1000, C: int = 192, expansion: int = 1, N: int = 4,
                 dilations: list[int] | None = None, mid_blurpool: bool = True,
                 max_gn_groups: int = 16, dropout: float = 0.3):
        super().__init__()
        self.C = C
        self.N = N
        self.mid_blurpool = mid_blurpool

        # stem: 7x7 conv stride=2 + BlurPool stride=2 (224 -> 112 -> 56)
        self.stem = nn.Conv2d(3, C, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_gn = nn.GroupNorm(gn_groups_for(C, max_gn_groups), C)
        self.stem_pool = BlurPool2d(C, stride=2)

        self.block = ECBlock(C, expansion=expansion, max_gn_groups=max_gn_groups)

        self.blur = BlurPool2d(C, stride=2)

        # 2-layer MLP head (4096x4096) with dropout
        self.fc1 = nn.Linear(C, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(4096, num_classes)

        # dilation schedule
        if dilations is None:
            dilations = [1, 1, 2, 1, 2, 3]  # pre-DS: 1,1,2  | post-DS: 1,2,3
            # dilations = [1, 1, 1, 1, 1, 1]  # pre-DS: 1,1,2  | post-DS: 1,2,3
        assert len(dilations) >= N, "Provide >= N dilations or adjust N"
        self.dilations = dilations[:N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_gn(x)
        x = self.stem_pool(x)
        for t in range(self.N):
            x = self.block(x, dilation=self.dilations[t])
            if self.mid_blurpool and t == (self.N // 2) - 1:
                x = self.blur(x)
        x = x.mean(dim=(2, 3))  # GAP
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.head(x)
        return x