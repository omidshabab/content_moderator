import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
import io
from typing import Dict, Union
from app.api.models import ContentFlag

class ImageAnalyzer:
    def __init__(self):
        # Load pre-trained MobileNet model - smaller and faster than ResNet
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        # Modify the classifier for our specific task
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(self.model.last_channel, 2)  # 2 classes: safe/unsafe
        )
        self.model.eval()
        
        # Standard normalization for pre-trained models
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load the state dict of your fine-tuned model
        # Make sure to have this file in your models directory
        try:
            state_dict = torch.load('models/ml/nsfw_mobilenet.pth', map_location='cpu')
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")

    def decode_image(self, image_data: Union[str, bytes]) -> Image.Image:
        try:
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Error decoding image: {str(e)}")

    def analyze(self, image_data: Union[str, bytes]) -> Dict:
        try:
            # Decode and transform image
            pil_image = self.decode_image(image_data)
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.softmax(output, dim=1)[0]
                
                safe_score = float(probabilities[0])
                nsfw_score = float(probabilities[1])
            
            # Define thresholds
            SAFE_THRESHOLD = 0.7
            EXPLICIT_THRESHOLD = 0.8
            
            # Determine safety and flags
            is_safe = safe_score >= SAFE_THRESHOLD
            flags = []
            
            if is_safe:
                flags.append(ContentFlag.SAFE)
            elif nsfw_score >= EXPLICIT_THRESHOLD:
                flags.append(ContentFlag.EXPLICIT)
            else:
                flags.append(ContentFlag.SENSITIVE)
            
            return {
                "is_safe": is_safe,
                "confidence_score": float(safe_score),
                "flags": flags,
                "details": {
                    "model_scores": {
                        "safe_score": safe_score,
                        "nsfw_score": nsfw_score
                    }
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error analyzing image: {str(e)}")

def create_analyzer():
    return ImageAnalyzer()