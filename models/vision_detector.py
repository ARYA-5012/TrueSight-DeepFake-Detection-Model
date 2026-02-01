from utils.preprocessing import preprocess_image
import os

class VisionDetector:
    def __init__(self, weights_path='mobilevit_xs_deepfake_detector.pth'):
        try:
            import torch
            import torch.nn as nn
            import timm
        except ImportError as e:
            print(f"Vision dependencies missing: {e}")
            self.model = None
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.weights_path = weights_path
        self.load_weights()

    def _build_model(self):
        try:
            import timm
            import torch
            # Create MobileViT model
            model = timm.create_model('mobilevit_xs', pretrained=True, num_classes=2)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Failed to create Vision Model: {e}")
            return None

    def load_weights(self):
        import torch
        if os.path.exists(self.weights_path):
            print(f"Loading Vision weights from {self.weights_path}")
            try:
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"WARNING: Vision weights not found at {self.weights_path}. Model will provide random predictions until trained.")

    def predict(self, image_path):
        import torch
        if self.model is None:
            return "Model not initialized"

        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            return "Error processing image"
            
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            fake_prob = probs[0][1].item()
            
        label = "FAKE" if fake_prob > 0.5 else "GENUINE"
        confidence = fake_prob if label == "FAKE" else 1 - fake_prob
        
        return {
            "label": label,
            "confidence": confidence,
            "raw_score": fake_prob
        }

    def train(self, data_dir, epochs=5):
        """
        Skeleton for training loop.
        """
        print("Training functionality to be implemented needs a valid dataset structure.")
