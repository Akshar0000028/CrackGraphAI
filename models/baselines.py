from __future__ import annotations

import segmentation_models_pytorch as smp


def build_unetplusplus():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def build_deeplabv3plus():
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def build_segformer():
    return smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
