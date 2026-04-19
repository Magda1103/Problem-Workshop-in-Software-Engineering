import torch
from src.model_utils.baseline_model import create_model


def test_model_forward_pass_shape():
    """
    Tests the neural network architecture by passing a dummy 5D tensor
    (Batch, Channels, Frames, Height, Width) and verifying that the output shape
    matches the expected number of action classes.
    """
    model = create_model(num_classes=9, backbone_type='resnet18')
    model.eval()

    dummy_input = torch.randn(1, 3, 10, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 9), f"Expected shape (1, 9), but got {output.shape}"