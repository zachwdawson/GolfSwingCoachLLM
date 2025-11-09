"""Model definition and loading utilities for GolfDBFrameClassifier."""
import logging

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.models.video import R3D_18_Weights
    from torchvision.models.video.resnet import BasicBlock
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    torchvision = None  # type: ignore
    R3D_18_Weights = None  # type: ignore
    BasicBlock = None  # type: ignore

logger = logging.getLogger(__name__)


def strip_temporal_stride(block: BasicBlock):
    """Strip temporal stride from a BasicBlock to maintain temporal resolution."""
    conv1 = block.conv1[0]

    if conv1.stride[0] == 2:
        new_stride = (1, conv1.stride[1], conv1.stride[2])
        conv1.stride = new_stride

        if block.downsample is not None:
            ds_conv = block.downsample[0]
            if isinstance(ds_conv, nn.Conv3d):
                ds_conv.stride = new_stride


class GolfDBFrameClassifier(nn.Module):
    """Frame classifier for golf swing events based on R3D-18 architecture."""

    def __init__(self, num_classes: int):
        super().__init__()

        base = torchvision.models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)

        strip_temporal_stride(base.layer2[0])
        strip_temporal_stride(base.layer3[0])
        strip_temporal_stride(base.layer4[0])

        for name, p in base.named_parameters():
            if name.split(".")[0] in ["stem", "layer1", "layer2"]:
                p.requires_grad = False  # Freeze these layers.

        # Just copy the convention from the base model.
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # 256 is sort of arbitrary but seems alright.
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Regularize...
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(dim=[-2, -1])  # Global average pooling.

        B, C, T = x.shape
        x = x.permute(0, 2, 1).reshape(B * T, C)
        # Flatten out the batches & times.

        logits = self.classifier(x)
        logits = logits.view(B, T, -1)

        return logits


def load_model(checkpoint_path: str, device: str) -> GolfDBFrameClassifier:
    """Load a trained model from checkpoint file."""
    try:
        logger.info(f"Loading model from {checkpoint_path}")
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        raise

