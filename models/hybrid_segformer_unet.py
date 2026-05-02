from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models as tv_models


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to match the spatial size of the skip connection
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TransformerBottleneck(nn.Module):
    """
    Fallback transformer bottleneck used when timm SegFormer is unavailable.
    Processes the deepest CNN feature map with self-attention.
    """

    def __init__(self, in_channels: int, embed_dim: int = 256, num_heads: int = 8, depth: int = 2):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B, N, C
        tokens = self.encoder(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.proj_out(x)


class ResNetEncoder(nn.Module):
    def __init__(self, name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        if name == "resnet34":
            backbone = tv_models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            channels = [64, 64, 128, 256, 512]
        elif name == "resnet50":
            backbone = tv_models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported CNN backbone: {name}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.channels = channels

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)          # stride 2  → H/2  × W/2
        x = self.pool(s0)          # stride 4  → H/4  × W/4
        s1 = self.layer1(x)        # stride 4  → H/4  × W/4
        s2 = self.layer2(s1)       # stride 8  → H/8  × W/8
        s3 = self.layer3(s2)       # stride 16 → H/16 × W/16
        s4 = self.layer4(s3)       # stride 32 → H/32 × W/32
        return [s0, s1, s2, s3, s4]


class HybridSegformerUNet(nn.Module):
    """
    Hybrid CNN + Transformer segmentation model.

    Architecture:
        - CNN branch  : ResNet34/50 encoder with 5 skip connections
        - Transformer : SegFormer B0 (timm) or TransformerBottleneck fallback
        - Fusion      : Concatenate deepest CNN feature + transformer output → 1×1 Conv
        - Decoder     : 4 U-Net decoder blocks with skip connections
        - Head        : 1×1 Conv → bilinear upsample to input resolution

    Output size equals input size (no spurious 2× upsampling).
    """

    def __init__(
        self,
        cnn_backbone: str = "resnet34",
        transformer_backbone: str = "mit_b0",
        classes: int = 1,
    ):
        super().__init__()
        self.cnn = ResNetEncoder(name=cnn_backbone, pretrained=True)
        transformer_module, transformer_channels, uses_timm = self._build_transformer(
            transformer_backbone, self.cnn.channels[-1]
        )
        self.transformer = transformer_module
        self.uses_timm = uses_timm

        # Fusion: deepest CNN feature (512) + transformer output → 512
        self.fusion = ConvBNReLU(self.cnn.channels[-1] + transformer_channels, 512)

        # Decoder blocks (bottom-up)
        self.dec4 = DecoderBlock(512, self.cnn.channels[3], 256)   # → H/16
        self.dec3 = DecoderBlock(256, self.cnn.channels[2], 128)   # → H/8
        self.dec2 = DecoderBlock(128, self.cnn.channels[1], 64)    # → H/4
        self.dec1 = DecoderBlock(64,  self.cnn.channels[0], 32)    # → H/2

        # Final prediction head
        self.head = nn.Conv2d(32, classes, kernel_size=1)

    @staticmethod
    def _build_transformer(backbone_name: str, in_channels: int):
        """Try to load a timm SegFormer backbone; fall back to TransformerBottleneck."""
        aliases = {
            "mit_b0": ["mit_b0", "mix_transformer_b0", "segformer_b0"],
            "mit_b1": ["mit_b1", "mix_transformer_b1", "segformer_b1"],
            "mit_b2": ["mit_b2", "mix_transformer_b2", "segformer_b2"],
            "mit_b3": ["mit_b3", "mix_transformer_b3", "segformer_b3"],
            "mit_b4": ["mit_b4", "mix_transformer_b4", "segformer_b4"],
            "mit_b5": ["mit_b5", "mix_transformer_b5", "segformer_b5"],
        }
        candidates = aliases.get(backbone_name, [backbone_name])
        last_error = None
        for candidate in candidates:
            try:
                module = timm.create_model(
                    candidate, pretrained=True, features_only=True, out_indices=(3,)
                )
                out_channels = module.feature_info.channels()[-1]
                print(f"[HybridSegformerUNet] Using timm backbone: {candidate}")
                return module, out_channels, True
            except Exception as err:
                last_error = err

        print(
            f"[HybridSegformerUNet] Warning: unable to load timm SegFormer backbones "
            f"{candidates}. Falling back to TransformerBottleneck. "
            f"Last error: {last_error}"
        )
        fallback = TransformerBottleneck(in_channels=in_channels, embed_dim=256, num_heads=8, depth=2)
        return fallback, fallback.out_channels, False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]   # (H, W) — remember for final upsample

        # ── Encoder ──────────────────────────────────────────────────────────
        skips = self.cnn(x)         # [s0, s1, s2, s3, s4]

        if self.uses_timm:
            tf = self.transformer(x)[0]          # timm: process full image
        else:
            tf = self.transformer(skips[-1])     # fallback: process deepest feature

        # Align transformer output to deepest CNN feature spatial size
        tf = F.interpolate(tf, size=skips[-1].shape[-2:], mode="bilinear", align_corners=False)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = self.fusion(torch.cat([skips[-1], tf], dim=1))

        # ── Decoder ───────────────────────────────────────────────────────────
        x = self.dec4(fused,  skips[3])   # H/16
        x = self.dec3(x,      skips[2])   # H/8
        x = self.dec2(x,      skips[1])   # H/4
        x = self.dec1(x,      skips[0])   # H/2

        # ── Head + final upsample to input resolution ─────────────────────────
        x = self.head(x)
        # Upsample from H/2 back to H (the decoder only reaches H/2 via s0)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
