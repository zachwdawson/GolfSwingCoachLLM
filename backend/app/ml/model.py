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


def load_model(checkpoint_path: str, device: str, num_classes: int = 9) -> GolfDBFrameClassifier:
    """
    Load a trained model from checkpoint file.
    
    Handles models saved with different module structures by loading only the
    state dict (weights) and creating a new model instance with the current class.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
        num_classes: Number of classes (default: 9 for 8 events + 1 no-event)
    
    Returns:
        Loaded model instance
    """
    try:
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Try to load just the state dict first (avoids module import issues)
        try:
            # Load with weights_only=True to avoid unpickling the model class
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint
        except Exception:
            # If weights_only fails, use custom pickle module with torch.load
            logger.info("weights_only failed, using custom pickle module")
            import pickle
            
            # Create a custom pickle module that maps AG_model to current module
            class CustomPickleModule:
                class Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'AG_model' and name == 'GolfDBFrameClassifier':
                            return GolfDBFrameClassifier
                        return super().find_class(module, name)
                
                Unpickler = Unpickler
                Pickler = pickle.Pickler
                loads = pickle.loads
                dumps = pickle.dumps
            
            # Try using pickle_module parameter (available in older PyTorch versions)
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=False,
                    pickle_module=CustomPickleModule
                )
            except TypeError:
                # pickle_module not supported, use manual unpickling
                logger.info("pickle_module not supported, using manual unpickling")
                # This is a fallback - we'll need to handle PyTorch's format manually
                # For now, raise an error with helpful message
                raise RuntimeError(
                    "Cannot load checkpoint with different module structure. "
                    "Please re-save the model with torch.save(model.state_dict(), path) "
                    "to save only the weights, or use a PyTorch version that supports pickle_module."
                )
            
            # Extract state dict from the loaded model
            if isinstance(checkpoint, nn.Module):
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict):
                # Check if it's a checkpoint dict with state_dict key
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Assume it's already a state dict
                    state_dict = checkpoint
            else:
                raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
        
        # Create a new model instance with the current class structure
        logger.info(f"Creating new model instance with {num_classes} classes")
        model = GolfDBFrameClassifier(num_classes=num_classes)
        
        # Load the state dict into the new model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        raise

