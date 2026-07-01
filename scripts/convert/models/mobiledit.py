"""
Self-contained MobileI2V model architecture for forward pass and ONNX export.

Vendored from https://github.com/hustvl/MobileI2V
Apache-2.0 License

Architecture: Mobiledit_300M_P1_D16
  - PatchEmbed: 2D patch embedding
  - 16x SanaBlock (7 cross + 1 vanila + 7 cross + 1 vanila)
  - Attention: LiteLA (linear attention with RoPE3D)
  - FFN: GLUMBConv
  - Condition: t2i_modulate (adaLN-single)
  - Timestep embedder + Flow score embedder + Caption embedder
  - T2IFinalLayer: final modulation + linear projection

All dependencies (timm helpers, einops patterns) are vendored inline.
No xformers, no triton, no checkpointing, no distributed.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Utility functions
# ============================================================

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def val2list(x, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x, min_len=1, idx_repeat=-1):
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(get_same_padding(ks) for ks in kernel_size)
    assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
    return kernel_size // 2


def auto_grad_checkpoint(module, *args, **kwargs):
    """Simplified: no checkpointing, just calls module directly."""
    return module(*args, **kwargs)


# ============================================================
# DropPath (from timm)
# ============================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, scale_factor=1.0, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)


# ============================================================
# PositionGetter3D and RoPE3D
# ============================================================

class PositionGetter3D(object):
    """Return 3D positions of patches."""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, t, h, w, device):
        key = (b, t, h, w, device)
        if key not in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))
            self.cache_positions[key] = (poses, max_poses)
        return self.cache_positions[key]


class RoPE3D(nn.Module):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1.4375, 2.5)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        key = (D, seq_len, device, dtype, interpolation_scale)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()
            sin = freqs.sin()
            self.cache[key] = (cos, sin)
        return self.cache[key]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = F.embedding(pos1d, cos)[:, None, :, :]
        sin = F.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2
        cos_t, sin_t = self.get_cos_sin(
            D, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t
        )
        cos_y, sin_y = self.get_cos_sin(
            D, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h
        )
        cos_x, sin_x = self.get_cos_sin(
            D, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w
        )
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens


# ============================================================
# build_act (simplified)
# ============================================================

def build_act(name=None, **kwargs):
    if name is None or name.lower() == "none":
        return None
    name = name.lower()
    if name == "silu" or name == "swish":
        return nn.SiLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(approximate="tanh")
    elif name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")


# ============================================================
# LayerNorm2d + build_norm (simplified)
# ============================================================

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def build_norm(name=None, num_features=None, affine=True, **kwargs):
    if name is None or name.lower() == "none":
        return None
    name = name.lower()
    if name == "bn2d":
        return nn.BatchNorm2d(num_features, affine=affine)
    elif name == "ln":
        return nn.LayerNorm(num_features, elementwise_affine=affine)
    elif name == "ln2d":
        return LayerNorm2d(num_features, elementwise_affine=affine)
    else:
        raise ValueError(f"Unsupported norm: {name}")


# ============================================================
# ConvLayer
# ============================================================

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding=None,
        use_bias=False,
        dropout=0.0,
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ============================================================
# GLUMBConv
# ============================================================

class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding=None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = build_act(act[1])

        self.inverted_conv = ConvLayer(
            in_features, hidden_features * 2, 1,
            use_bias=use_bias[0], norm=norm[0], act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2, hidden_features * 2,
            kernel_size, stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features, out_feature, 1,
            use_bias=use_bias[2], norm=norm[2], act=act[2],
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        x = self.point_conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


# ============================================================
# Mlp (from timm, vendored inline)
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DWMlp(Mlp):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, bias=True, drop=0.0,
                 kernel_size=3, stride=1, dilation=1, padding=None):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation
        self.conv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=hidden_features,
            bias=bias,
        )

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        if H is None or W is None:
            H = W = int(N ** 0.5)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = x.reshape(B, H, W, self.hidden_features).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, self.hidden_features, N).permute(0, 2, 1)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ============================================================
# t2i_modulate
# ============================================================

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================
# MultiHeadCrossAttention (no xformers, vanilla fallback)
# ============================================================

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(B, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(q.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)
        x = x.contiguous().view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
# Vanilla Attention (no timm dependency, with RoPE3D)
# ============================================================

class Attention(nn.Module):
    """Vanilla multi-head self-attention with RoPE3D."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=False, **block_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

        self.rope = RoPE3D(interpolation_scale_thw=(1, 1.4375, 2.5))
        self.position_getter = PositionGetter3D()

        if qk_norm:
            self.q_norm = RMSNorm(dim, scale_factor=1.0, eps=1e-5)
            self.k_norm = RMSNorm(dim, scale_factor=1.0, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _compute_rope_positions(self, q, T):
        """Compute RoPE positions from token count and temporal factor."""
        B, H, N, D = q.shape
        S = N // T
        # Estimate H_spatial, W_spatial from spatial token count
        h_spatial = int(S ** 0.5)
        w_spatial = S // h_spatial
        while h_spatial * w_spatial < S:
            w_spatial += 1
        while h_spatial * w_spatial > S:
            w_spatial -= 1
        pos_thw = self.position_getter(B, t=T, h=h_spatial, w=w_spatial, device=q.device)
        return pos_thw

    def forward(self, x, HW=None, T=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE3D
        if T is None:
            T = 3  # default fallback
        pos_thw = self._compute_rope_positions(q, T)
        q = self.rope(q, pos_thw)
        k = self.rope(k, pos_thw)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
# LiteLA (Lightweight Linear Attention with RoPE3D)
# ============================================================

class LiteLA(nn.Module):
    """Lightweight linear attention with 3D RoPE."""

    PAD_VAL = 1

    def __init__(self, in_dim, out_dim, heads=None, heads_ratio=1.0,
                 dim=32, eps=1e-15, use_bias=False, qk_norm=False, norm_eps=1e-5):
        super().__init__()
        heads = heads or int(out_dim // dim * heads_ratio)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps

        self.qkv = nn.Linear(in_dim, out_dim * 3, bias=use_bias)
        self.proj = nn.Linear(out_dim, out_dim)

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rope = RoPE3D()
        self.position_getter = PositionGetter3D()

    def _compute_rope_positions(self, q, T):
        """Compute RoPE positions from tensor shape and temporal factor."""
        B, h, N, D = q.shape
        S = N // T
        h_spatial = int(S ** 0.5)
        w_spatial = S // h_spatial
        while h_spatial * w_spatial < S:
            w_spatial += 1
        while h_spatial * w_spatial > S:
            w_spatial -= 1
        pos_thw = self.position_getter(B, t=T, h=h_spatial, w=w_spatial, device=q.device)
        return pos_thw

    def attn_matmul(self, q, k, v):
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)  # (B, h, h_d, N) @ (B, h, N, h_d) -> (B, h, h_d, h_d)
        out = torch.matmul(vk, q)  # (B, h, h_d, h_d) @ (B, h, h_d, N) -> (B, h, h_d, N)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out

    def forward(self, x, mask=None, HW=None, block_id=None, T=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, C, N)
        v = v.transpose(-1, -2)  # (B, C, N)

        q = q.reshape(B, C // self.dim, self.dim, N)          # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)          # (B, h, h_d, N)

        q = q.transpose(-1, -2)  # (B, h, N, h_d)

        # Apply RoPE3D
        if T is None:
            T = 3  # default fallback
        pos_thw = self._compute_rope_positions(q, T)
        q = self.rope(q, pos_thw)
        k = self.rope(k, pos_thw)

        q = q.transpose(-1, -2)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


# ============================================================
# FlashAttention (vanilla fallback, no xformers)
# ============================================================

class FlashAttention(nn.Module):
    """Multi-head attention using PyTorch's scaled_dot_product_attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=False, **block_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, mask=None, HW=None, block_id=None, T=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, self.head_dim).to(dtype)
        k = k.reshape(B, N, self.num_heads, self.head_dim).to(dtype)
        v = v.reshape(B, N, self.num_heads, self.head_dim).to(dtype)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)

        x = x.contiguous().view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
            x = x.clip(-65504, 65504)

        return x


# ============================================================
# PatchEmbed
# ============================================================

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_height=224, img_width=224, patch_size=16,
                 in_chans=3, embed_dim=768, kernel_size=None,
                 padding=0, norm_layer=None, flatten=True, bias=True):
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_2tuple(patch_size)
        self.img_size = (img_height, img_width)
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // patch_size[0], self.img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=patch_size, padding=padding, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.flatten:
            x = self.proj(x).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x)
        x = self.norm(x)
        return x


# ============================================================
# TimestepEmbedder
# ============================================================

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, :, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


# ============================================================
# CaptionEmbedder
# ============================================================

class CaptionEmbedder(nn.Module):
    """Embeds text captions with classifier-free guidance dropout."""

    def __init__(self, in_channels, hidden_size, uncond_prob,
                 act_layer=nn.GELU(approximate="tanh"), token_num=120):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size,
            out_features=hidden_size, act_layer=act_layer, drop=0,
        )
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).to(caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


# ============================================================
# T2IFinalLayer
# ============================================================

class T2IFinalLayer(nn.Module):
    """Final layer of the diffusion transformer with modulation."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, :, None]).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================
# SanaBlock_cross
# ============================================================

class SanaBlock_cross(nn.Module):
    """
    Transformer block with self-attention + MLP.
    Uses adaLN-single (t2i_modulate) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if attn_type == "flash":
            self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, **block_kwargs)
        elif attn_type == "linear":
            linear_head_dim = 72
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "vanilla":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu, drop=0,
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu, drop=0,
            )
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, flow_score, mask=None, H=None, W=None, T=None, S=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, N, 6, -1) + flow_score.reshape(B, N, 6, -1)
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        # Self-attention with adaLN modulation
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x_s = self.attn(x_m, HW=(H, W) if H is not None and W is not None else None, T=T)
        x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # MLP with adaLN modulation and temporal rearrangement
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_mlp = self.mlp(x_m, H, W)
        x_mlp = rearrange(x_mlp, "(B T) S C -> B (T S) C", T=T, S=S)
        x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


# ============================================================
# SanaBlock_vanila
# ============================================================

class SanaBlock_vanila(nn.Module):
    """
    Transformer block with vanilla self-attention + MLP.
    Uses adaLN-single (t2i_modulate) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu, drop=0,
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu, drop=0,
            )
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, flow_score, mask=None, H=None, W=None, T=None, S=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, N, 6, -1) + flow_score.reshape(B, N, 6, -1)
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        # Self-attention with adaLN modulation
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x_s = self.attn(x_m, HW=(H, W) if H is not None and W is not None else None, T=T)
        x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # MLP with adaLN modulation and temporal rearrangement
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_mlp = self.mlp(x_m, H, W)
        x_mlp = rearrange(x_mlp, "(B T) S C -> B (T S) C", T=T, S=S)
        x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


# ============================================================
# Position embedding helpers
# ============================================================

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0,
                            pe_interpolation=1.0, base_size=(16, 16)):
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


# ============================================================
# Main Mobiledit Model
# ============================================================

class Mobiledit(nn.Module):
    """
    MobileI2V diffusion model with transformer backbone.

    Input:  (B, C, T, H, W) latent video
    Output: (B, C_out, T, H, W) predicted noise/denoised latents
    """

    def __init__(
        self,
        input_height=32,
        input_width=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=120,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="flash",
        ffn_type="mlp",
        use_pe=True,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.y_norm = y_norm
        self.input_size = (17, input_height, input_width)
        self.hidden_size = hidden_size

        num_patches = np.prod([self.input_size[i] // 1 for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = self.input_size[0] // 1
        self.num_spatial = num_patches // self.num_temporal

        kernel_size = patch_embed_kernel or patch_size

        self.x_embedder = PatchEmbed(
            input_height, input_width, patch_size, in_channels, hidden_size,
            kernel_size=kernel_size, bias=True,
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.flow_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = (input_height // self.patch_size, input_width // self.patch_size)

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.flow_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth + 2)]

        # Block arrangement: 7 cross + 1 vanila + 7 cross + 1 vanila
        self.blocks = nn.ModuleList(
            [
                SanaBlock_cross(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_height // patch_size, input_width // patch_size),
                    qk_norm=qk_norm, attn_type=attn_type, ffn_type=ffn_type,
                    mlp_acts=mlp_acts, linear_head_dim=linear_head_dim,
                )
                for i in range(7)
            ]
        )
        for _ in range(1):
            self.blocks.append(
                SanaBlock_vanila(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[14],
                    input_size=(input_height // patch_size, input_width // patch_size),
                    attn_type=attn_type, ffn_type=ffn_type,
                    mlp_acts=mlp_acts, linear_head_dim=linear_head_dim,
                )
            )
        for i in range(7):
            self.blocks.append(
                SanaBlock_cross(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_height // patch_size, input_width // patch_size),
                    qk_norm=qk_norm, attn_type=attn_type, ffn_type=ffn_type,
                    mlp_acts=mlp_acts, linear_head_dim=linear_head_dim,
                )
            )
        for _ in range(1):
            self.blocks.append(
                SanaBlock_vanila(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[14],
                    input_size=(input_height // patch_size, input_width // patch_size),
                    attn_type=attn_type, ffn_type=ffn_type,
                    mlp_acts=mlp_acts, linear_head_dim=linear_head_dim,
                )
            )

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size != 0:
            T += self.patch_size - T % self.patch_size
        if H % self.patch_size != 0:
            H += self.patch_size - H % self.patch_size
        if W % self.patch_size != 0:
            W += self.patch_size - W % self.patch_size
        T = T // self.patch_size
        H = H // self.patch_size
        W = W // self.patch_size
        return T, H, W

    def forward(self, x, timestep, guide_image, y, cond_mask, flow_score,
                mask=None, data_info=None, **kwargs):
        """
        Forward pass of Mobiledit.

        Args:
            x: (B, C, T, H, W) latent video
            timestep: (B, N) or (B,) timestep values
            guide_image: (B, C, 1, H, W) first frame (guide) image
            y: (B, 1, L, C) text embeddings
            cond_mask: (B,) conditioning mask
            flow_score: (B,) flow score values
            mask: optional attention mask
            data_info: optional dict with extra info
        """
        B = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W

        # Patch embed: (B, C, T, H, W) -> (B*T, C, H, W) -> (B*T, S, D) -> (B, T*S, D)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.x_embedder(x)  # (B*T, S, D)
        x = rearrange(x, "(B T) S C -> B (T S) C", B=B, T=T, S=S)

        # Timestep and flow score conditioning
        num_patches = T * S
        timestep = timestep.unsqueeze(1).repeat(1, num_patches)
        cond_mask = cond_mask.unsqueeze(1).repeat(1, num_patches)
        timestep = torch.min(timestep, (1.0 - cond_mask) * 1000)
        timestep = timestep / 1000.0
        t = self.t_embedder(timestep.to(x.dtype))  # (B, N, D)
        t0 = self.t_block(t)  # (B, N, 6*D)

        flow_score = flow_score.unsqueeze(1).repeat(1, num_patches)
        flow_score_emb = self.flow_embedder(flow_score.to(x.dtype))  # (B, N, D)
        flow_score_emb = self.flow_block(flow_score_emb)  # (B, N, 6*D)

        # Text embedding
        y = self.y_embedder(y, self.training)  # (B, 1, L, D)
        if self.y_norm:
            y = self.attention_y_norm(y)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # Transformer blocks
        for block in self.blocks:
            x = auto_grad_checkpoint(
                block, x, y, t0, flow_score_emb, y_lens, H, W, T, S,
            )

        # Final layer
        x = self.final_layer(x, t)  # (B, N, patch_size^2 * out_channels)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Convert patch tokens back to latent video.

        Args:
            x: (B, N_t*N_h*N_w, patch_size^2 * C_out)
            N_t, N_h, N_w: number of patches per dimension
            R_t, R_h, R_w: actual temporal, height, width (before padding)
        """
        T_p = H_p = W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t, N_h=N_h, N_w=N_w,
            T_p=T_p, H_p=H_p, W_p=W_p,
            C_out=self.out_channels,
        )
        # Unpad to original size
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size, grid_size[1] // self.patch_size),
            pe_interpolation=1,
            base_size=self.base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size,
            scale=1.0,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_pe:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.x_embedder.num_patches ** 0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize flow embedding MLP:
        nn.init.normal_(self.flow_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.flow_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.flow_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# ============================================================
# Factory function
# ============================================================

def mobiledit_300m_P1_D16(**kwargs):
    """Create Mobiledit-300M with patch_size=1 and depth=16."""
    return Mobiledit(
        depth=16, hidden_size=1152, patch_size=1, num_heads=16,
        in_channels=128,
        pred_sigma=False,
        caption_channels=896,
        model_max_length=300,
        **kwargs,
    )


# ============================================================
# ONNX-friendly wrapper
# ============================================================

class MobileditONNXWrapper(nn.Module):
    """ONNX-friendly wrapper around Mobiledit.

    Exposes a clean forward(latent, text_emb, timestep) interface.
    guide_image and flow_score are injected as constants.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent, text_emb, timestep):
        """
        Args:
            latent: (B, C, T, H, W) latent video tensor
            text_emb: (B, 1, L, C) text embeddings
            timestep: (B,) or (B, 1) diffusion timestep
        Returns:
            (B, C_out, T, H, W) predicted denoised latent
        """
        B = latent.shape[0]

        # guide_image: use first frame of latent
        guide_image = latent[:, :, :1, :, :]

        # cond_mask: always 1.0 (fully conditioned)
        cond_mask = torch.ones(B, device=latent.device, dtype=latent.dtype)

        # flow_score: default 2.0
        flow_score = torch.full((B,), 2.0, device=latent.device, dtype=latent.dtype)

        return self.model(latent, timestep, guide_image, text_emb, cond_mask, flow_score)
