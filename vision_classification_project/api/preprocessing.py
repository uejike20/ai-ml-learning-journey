# api/preprocessing.py
from torchvision import transforms
from PIL import Image
import io
import torch # Make sure to import torch

# These should be the same as your validation transforms from the notebook
# Make sure image size (224) and normalization stats are correct for your model (ResNet18 default)
def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Transforms image bytes into a tensor suitable for the model.
    """
    image_size = 224
    # These are ImageNet normalization stats, commonly used for ResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    return transform(image).unsqueeze(0) # Add batch dimension