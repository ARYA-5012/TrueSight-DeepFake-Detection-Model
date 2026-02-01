import os
import numpy as np
from utils.preprocessing import preprocess_audio

class AudioDetector:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.load_model()

    def load_model(self):
        """
        Download and load the pre-trained model from KaggleHub.
        """
        try:
            import tensorflow as tf
            import kagglehub
        except ImportError as e:
            print(f"Audio dependencies missing: {e}")
            return
        print("Downloading/Loading Audio Model...")
        try:
            # Download model
            path = kagglehub.model_download('vivekshinde777/audio_deepfake_cnn-and-bilstm_model/TensorFlow2/default/1')
            self.model_path = os.path.join(path, 'updated_model.h5')
            
            if not os.path.exists(self.model_path):
                 # Fallback: sometimes the file name might be different or path structure differs
                 # Searching for .h5 file in the downloaded path
                 for root, dirs, files in os.walk(path):
                     for file in files:
                         if file.endswith('.h5'):
                             self.model_path = os.path.join(root, file)
                             break
            
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Audio Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Audio Model: {e}")

    def predict(self, audio_path):
        """
        Predict if the audio is Real or Fake.
        """
        if self.model is None:
            return "Model not loaded"
            
        features = preprocess_audio(audio_path)
        if features is None:
            return "Error processing file"
            
        # Reshape for model input: (batch, n_mfcc, time, channels)
        # Expected shape based on original code: (1, 40, 500, 1)
        # Check current shape
        # features shape is (40, 500)
        
        input_data = features.reshape(1, 40, 500, 1)
        
        prediction = self.model.predict(input_data)
        probability = prediction[0][0]
        
        # Original code: > 0.5 means Fake (1)
        label = "FAKE" if probability > 0.5 else "GENUINE"
        confidence = probability if label == "FAKE" else 1 - probability
        
        return {
            "label": label,
            "confidence": float(confidence),
            "raw_score": float(probability)
        }
