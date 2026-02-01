import numpy as np
from PIL import Image

def preprocess_audio(file_path, sr=16000, max_length=500):
    """
    Load audio and extract MFCC features.
    """
    try:
        import librosa
    except ImportError as e:
        print(f"Audio preprocessing dependency missing: {e}")
        return None

    try:
        audio, _ = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Pad or trim
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
            
        return mfccs
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        return None

def preprocess_image(file_path, img_size=(224, 224)):
    """
    Load and preprocess image for model inference.
    """
    try:
        import torchvision.transforms as transforms
    except ImportError as e:
        print(f"Image preprocessing dependency missing: {e}")
        return None

    try:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None
