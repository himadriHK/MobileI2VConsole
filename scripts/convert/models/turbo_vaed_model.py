"""
turbo_vaed_model.py — Vendored Turbo-VAED Decoder model for ONNX export.

Self-contained implementation of all Turbo-VAED custom blocks, extracted from:
  https://github.com/hustvl/Turbo-VAED/blob/main/diffusers_vae/src/diffusers/models/autoencoders/autoencoder_kl_turbo_vaed.py

Apache-2.0 License. Original work by HUST Vision Lab / HuggingFace.

Usage:
    from models.turbo_vaed_model import build_turbo_vaed_decoder

    decoder = build_turbo_vaed_decoder(config_dict)
    state_dict = torch.load("Turbo-VAED-LTX.pth", map_location="cpu")
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


# ============================================================
# Vendored utility classes (from diffusers_vae)
# ============================================================

class RMSNorm(nn.Module):
    """RMS Normalization as used by Turbo-VAED."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x.to(input_dtype)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "swish" or name == "silu":
        return nn.SiLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


# ============================================================
# Custom Conv Blocks
# ============================================================

class TurboVAEDConv2dSplitUpsampler(nn.Module):
    """2D pixel-shuffle upsampler (spatial only, no temporal dim)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        height_pad = self.kernel_size[0] // 2
        width_pad = self.kernel_size[1] // 2
        padding = (height_pad, width_pad)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[0])
        return hidden_states


class TurboVAEDConv2dUpsampler(nn.Module):
    """2D pixel-shuffle upsampler (applied frame-by-frame along temporal dim)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels = in_channels

        height_pad = self.kernel_size[0] // 2
        width_pad = self.kernel_size[1] // 2
        padding = (height_pad, width_pad)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, C, T, H, W]
        batch_size, channels, time_steps, height, width = hidden_states.shape
        # Flatten batch+time for 2D conv
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size * time_steps, channels, height, width)
        hidden_states = self.conv(hidden_states)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[0])
        _, _, output_height, output_width = hidden_states.shape
        output_channels = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(batch_size, time_steps, output_channels, output_height, output_width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        return hidden_states


class TurboVAEDCausalConv3d(nn.Module):
    """3D causal or non-causal convolution used in Turbo-VAED."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        is_causal: bool = True,
    ):
        super().__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)  # temporal padding done manually

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]
        if self.is_causal:
            if time_kernel_size > 1:
                pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
                hidden_states = torch.cat([pad_left, hidden_states], dim=2)
        else:
            if time_kernel_size > 1:
                pad_count = (time_kernel_size - 1) // 2
                pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_count, 1, 1))
                pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad_count, 1, 1))
                hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class TurboVAEDCausalDepthwiseSeperableConv3d(nn.Module):
    """Depthwise-separable 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",
        is_causal: bool = True,
    ):
        super().__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.depthwise_conv = nn.Conv3d(
            in_channels, in_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=in_channels,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.pointwise_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]
        if time_kernel_size > 1:
            pad_count = (time_kernel_size - 1) // 2
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_count, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad_count, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.pointwise_conv(hidden_states)
        return hidden_states


# ============================================================
# TurboVAEDUpsampler3d — decoupled 3D pixel shuffle
# ============================================================

class TurboVAEDUpsampler3d(nn.Module):
    """Decoupled 3D pixel shuffle upsampler: temporal + spatial."""

    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        is_causal: bool = True,
        residual: bool = False,
        upscale_factor: int = 1,
        padding_mode: str = "zeros",
        is_video_dc_ae: bool = False,
    ):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor
        self.is_video_dc_ae = is_video_dc_ae

        out_channels = (in_channels * stride[0] * stride[1] * stride[2]) // upscale_factor

        self.conv = TurboVAEDCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = self.conv(hidden_states)

        # Step 1: temporal upsampling via reshape+permute
        hidden_states = hidden_states.reshape(batch_size, -1, self.stride[0], num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5)
        hidden_states = hidden_states.reshape(batch_size, -1, num_frames * self.stride[0], height, width)

        # Step 2: spatial 2D pixel shuffle
        upsampled_frames = num_frames * self.stride[0]
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.reshape(batch_size * upsampled_frames, -1, height, width)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[1])

        # Step 3: reshape back
        _, c, h, w = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, upsampled_frames, c, h, w)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        # Step 4: remove temporal padding
        if not self.is_video_dc_ae:
            hidden_states = hidden_states[:, :, self.stride[0] - 1:]

        return hidden_states


# ============================================================
# TurboVAEDResnetBlock3d
# ============================================================

class TurboVAEDResnetBlock3d(nn.Module):
    """3D ResNet block with optional depthwise separable conv and timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        is_upsampler_modified: bool = False,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.nonlinearity = get_activation(non_linearity)
        conv_op = TurboVAEDCausalDepthwiseSeperableConv3d if is_dw_conv else TurboVAEDCausalConv3d
        kernel_size = dw_kernel_size if is_dw_conv else 3

        self.is_upsampler_modified = is_upsampler_modified
        self.replace_nonlinearity = get_activation("relu")

        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.conv1 = conv_op(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, is_causal=is_causal,
        )

        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv_op(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, is_causal=is_causal,
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = conv_op(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, is_causal=is_causal,
            )

        # Noise injection (unused for inference, kept for compat)
        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        # Timestep conditioning (unused for inference, kept for compat)
        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels ** 0.5)

    def forward(
        self, hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        inputs = hidden_states

        hidden_states = self.norm1(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.scale_shift_table is not None and temb is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        if self.is_upsampler_modified:
            hidden_states = self.replace_nonlinearity(hidden_states)
        else:
            hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.per_channel_scale1 is not None and generator is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.scale_shift_table is not None and temb is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.per_channel_scale2 is not None and generator is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


# ============================================================
# TurboVAEDMidBlock3d
# ============================================================

class TurboVAEDMidBlock3d(nn.Module):
    """Middle block with multiple ResNet layers."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                TurboVAEDResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        return hidden_states


# ============================================================
# TurboVAEDUpBlock3d
# ============================================================

class TurboVAEDUpBlock3d(nn.Module):
    """Up block with optional spatio-temporal upsampling and ResNet layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        upsample_residual: bool = False,
        upscale_factor: int = 1,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
        spatio_only: bool = False,
        is_video_dc_ae: bool = False,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = TurboVAEDResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                is_causal=is_causal,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                is_dw_conv=is_dw_conv,
                dw_kernel_size=dw_kernel_size,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            stride_up = (2, 2, 2) if not spatio_only else (1, 2, 2)
            self.upsamplers = nn.ModuleList([
                TurboVAEDUpsampler3d(
                    out_channels * upscale_factor,
                    stride=stride_up,
                    is_causal=is_causal,
                    residual=upsample_residual,
                    upscale_factor=upscale_factor,
                    is_video_dc_ae=is_video_dc_ae,
                )
            ])

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                TurboVAEDResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                    is_upsampler_modified=spatio_temporal_scale,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator)
        return hidden_states


# ============================================================
# TurboVAEDDecoder3d
# ============================================================

class TurboVAEDDecoder3d(nn.Module):
    """
    The Turbo-VAED decoder.

    Args are designed to be compatible with the config dict from the
    AutoencoderKLTurboVAED class.  The `build_turbo_vaed_decoder()` factory
    performs the key name mapping from the diffusers-style config.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: Tuple[bool, ...] = (False, False, False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: Tuple[bool, ...] = (False, False, False, False),
        upsample_factor: Tuple[int, ...] = (1, 1, 1, 1),
        decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, False, False),
        decoder_dw_kernel_size: int = 3,
        spatio_only: Tuple[bool, ...] = (False, False, False, False),
        upsampling: bool = False,
        is_video_dc_ae: bool = False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.upsampling = upsampling

        # Reverse for decoder (built bottom-up)
        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        decoder_is_dw_conv = tuple(reversed(decoder_is_dw_conv))
        spatio_only = tuple(reversed(spatio_only))

        output_channel = block_out_channels[0]

        # Conv in
        self.conv_in = TurboVAEDCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1, is_causal=is_causal,
        )

        # Mid block
        self.mid_block = TurboVAEDMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            is_dw_conv=decoder_is_dw_conv[0],
            dw_kernel_size=decoder_dw_kernel_size,
        )

        # Up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            up_block = TurboVAEDUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_causal=is_causal,
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                is_dw_conv=decoder_is_dw_conv[i + 1],
                dw_kernel_size=decoder_dw_kernel_size,
                spatio_only=spatio_only[i],
                is_video_dc_ae=is_video_dc_ae,
            )
            self.up_blocks.append(up_block)

        # Final 2D upsamplers (pixel shuffle)
        if self.patch_size >= 2:
            self.norm_up_1 = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
            self.upsampler2d_1 = TurboVAEDConv2dSplitUpsampler(
                in_channels=output_channel, kernel_size=3, stride=2,
            )
            output_channel = output_channel // (2 * 2)

        if self.patch_size >= 4:
            self.norm_up_2 = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
            self.upsampler2d_2 = TurboVAEDConv2dUpsampler(
                in_channels=output_channel, kernel_size=3, stride=2,
            )
            output_channel = output_channel // (2 * 2)

        # Out norm
        if self.patch_size == 1:
            self.norm_out = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)

        self.conv_act = nn.SiLU()
        self.conv_out = TurboVAEDCausalConv3d(
            in_channels=output_channel, out_channels=self.out_channels, kernel_size=3, stride=1, is_causal=is_causal,
        )

    def forward(
        self, hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states: [B, C, T, H, W]
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.mid_block(hidden_states, temb)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb)

        # Final 2D spatial upsampling
        if self.patch_size >= 2:
            hidden_states = self.norm_up_1(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            hidden_states = self.conv_act(hidden_states)
            hidden_states_array = []
            for t in range(hidden_states.shape[2]):
                h = self.upsampler2d_1(hidden_states[:, :, t, :, :])
                hidden_states_array.append(h)
            hidden_states = torch.stack(hidden_states_array, dim=2)

        if self.patch_size >= 4:
            hidden_states = self.norm_up_2(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            hidden_states = self.conv_act(hidden_states)
            hidden_states = self.upsampler2d_2(hidden_states)

        if self.patch_size == 1:
            hidden_states = self.norm_out(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        else:
            variance = hidden_states.pow(2).mean(1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + 1e-8)

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


# ============================================================
# Factory: build decoder from AutoencoderKLTurboVAED config dict
# ============================================================

# Maps from AutoencoderKLTurboVAED config key → TurboVAEDDecoder3d arg
_DECODER_CONFIG_KEY_MAP = {
    "latent_channels": "in_channels",
    "out_channels": "out_channels",
    "decoder_block_out_channels": "block_out_channels",
    "decoder_layers_per_block": "layers_per_block",
    "patch_size": "patch_size",
    "patch_size_t": "patch_size_t",
    "resnet_norm_eps": "resnet_norm_eps",
    "decoder_causal": "is_causal",
    "timestep_conditioning": "timestep_conditioning",
    "decoder_inject_noise": "inject_noise",
    "upsample_residual": "upsample_residual",
    "upsample_factor": "upsample_factor",
    "decoder_is_dw_conv": "decoder_is_dw_conv",
    "decoder_dw_kernel_size": "decoder_dw_kernel_size",
    "decoder_spatio_only": "spatio_only",
    "is_video_dc_ae": "is_video_dc_ae",
}


def build_turbo_vaed_decoder(config: dict) -> TurboVAEDDecoder3d:
    """
    Build a TurboVAEDDecoder3d from an AutoencoderKLTurboVAED config dict
    (as found in the config JSON files at
    https://github.com/hustvl/Turbo-VAED/tree/main/configs).

    The config dict contains keys like `latent_channels`, `decoder_block_out_channels`,
    etc. which are mapped to the TurboVAEDDecoder3d constructor arguments.

    Args:
        config: Config dictionary from a Turbo-VAED JSON config file.

    Returns:
        A TurboVAEDDecoder3d instance (weights NOT loaded).
    """
    decoder_kwargs = {}

    for config_key, decoder_key in _DECODER_CONFIG_KEY_MAP.items():
        if config_key in config:
            decoder_kwargs[decoder_key] = config[config_key]

    # Handle spatio_temporal_scaling: use decoder_spatio_temporal_scaling if present,
    # otherwise fall back to spatio_temporal_scaling
    if "decoder_spatio_temporal_scaling" in config:
        decoder_kwargs["spatio_temporal_scaling"] = config["decoder_spatio_temporal_scaling"]
    elif "spatio_temporal_scaling" in config:
        decoder_kwargs["spatio_temporal_scaling"] = config["spatio_temporal_scaling"]

    # Set defaults for keys that might be missing
    decoder_kwargs.setdefault("in_channels", 128)
    decoder_kwargs.setdefault("out_channels", 3)
    decoder_kwargs.setdefault("block_out_channels", (128, 256, 512, 512))
    decoder_kwargs.setdefault("spatio_temporal_scaling", (True, True, True, False))
    decoder_kwargs.setdefault("layers_per_block", (4, 3, 3, 3, 4))
    decoder_kwargs.setdefault("patch_size", 4)
    decoder_kwargs.setdefault("is_causal", False)
    decoder_kwargs.setdefault("timestep_conditioning", False)
    decoder_kwargs.setdefault("decoder_is_dw_conv", (False, False, False, False, False))
    decoder_kwargs.setdefault("decoder_dw_kernel_size", 3)
    decoder_kwargs.setdefault("resnet_norm_eps", 1e-6)
    decoder_kwargs.setdefault("inject_noise", (False, False, False, False, False))
    decoder_kwargs.setdefault("upsample_residual", (False, False, False, False))
    decoder_kwargs.setdefault("upsample_factor", (1, 1, 1, 1))
    decoder_kwargs.setdefault("spatio_only", (False, False, False, False))
    decoder_kwargs.setdefault("is_video_dc_ae", False)

    return TurboVAEDDecoder3d(**decoder_kwargs)
