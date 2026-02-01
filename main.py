import argparse
import os
import sys

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from models.audio_detector import AudioDetector
from models.vision_detector import VisionDetector

def main():
    parser = argparse.ArgumentParser(description="TrueSight Deepfake Detection")
    parser.add_argument('--file', type=str, required=True, help="Path to the image or audio file to analyze")
    args = parser.parse_args()
    
    file_path = args.file
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.wav', '.mp3', '.flac']:
        print(f"Detected Audio File: {file_path}")
        detector = AudioDetector()
        result = detector.predict(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"Detected Image File: {file_path}")
        detector = VisionDetector()
        result = detector.predict(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        sys.exit(1)
        
    print("-" * 30)
    print("ANALYSIS RESULT")
    print("-" * 30)
    print(f"File: {os.path.basename(file_path)}")
    
    if isinstance(result, dict):
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Score:  {result['raw_score']:.4f}")
    else:
        print(f"Status: {result}")
        
    print("-" * 30)

if __name__ == "__main__":
    main()
