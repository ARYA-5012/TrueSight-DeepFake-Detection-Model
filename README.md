# 🛡️ TrueSight: Multimodal Deepfake Detection Framework

> **Deepfake detection across Audio and Video modalities.**

TrueSight is a robust framework designed to fight misinformation by detecting deepfake content. It leverages a combination of **Convolutional Neural Networks (CNNs)**, **Bi-Directional LSTMs**, and **Vision Transformers (MobileViT)** to analyze both audio and visual data.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-TensorFlow%20%7C%20PyTorch-orange)

## 🚀 Key Features

*   **Audio Detection**: Uses a CNN + BiLSTM architecture to detect AI-generated speech (TTS/Voice Cloning). Features are extracted using MFCCs.
*   **Vision Detection**: Utilizes MobileViT (lightweight Vision Transformer) for detecting face manipulations in images.
*   **Unified Interface**: Simple CLI (`main.py`) to handle both image and audio files transparently.
*   **Lazy Loading**: Optimized for performance; loads only the necessary models based on the input file type.

## 📁 Project Structure

```
TrueSight/
├── main.py                 # 🏁 Entry point for the CLI
├── requirements.txt        # 📦 Project dependencies
├── models/                 # 🧠 Model definitions
│   ├── audio_detector.py   # Wrapper for Audio Model (auto-downloads weights)
│   └── vision_detector.py  # Wrapper for MobileViT Vision Model
├── utils/                  # 🛠️ Utility functions
│   └── preprocessing.py    # MFCC extraction & Image normalization
└── legacy/                 # 🏛️ Original research notebooks & scripts
```

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/TrueSight.git
    cd TrueSight
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate

    pip install -r requirements.txt
    ```

## 🎯 Usage

Run the main script with the `--file` argument. The system auto-detects the file type.

### Audio Detection
The audio model will **automatically download** from KaggleHub on the first run.
```bash
python main.py --file samples/fake_voice.wav
```
*Supported formats: .wav, .mp3, .flac*

### Image Detection
> **Note**: The Vision model architecture is set up, but requires training or pre-trained weights (`mobilevit_xs_deepfake_detector.pth`).
```bash
python main.py --file samples/suspected_face.jpg
```
*Supported formats: .jpg, .png, .jpeg, .bmp*

### Example Output
```text
------------------------------
ANALYSIS RESULT
------------------------------
File: fake_voice.wav
Prediction: FAKE
Confidence: 0.9852
Raw Score:  0.9852
------------------------------
```

## 🏗️ Model Details

### 🎤 Audio Model
-   **Architecture**: 2D-CNN for feature extraction followed by Bi-Directional LSTM for temporal sequence analysis.
-   **Input**: MFCC (Mel-frequency cepstral coefficients).
-   **Source**: [KaggleHub - Audio Deepfake Model](https://www.kaggle.com/models/vivekshinde777/audio_deepfake_cnn-and-bilstm_model)

### 👁️ Vision Model
-   **Architecture**: MobileViT-XS (Lightweight Vision Transformer).
-   **Input**: 224x224 RGB Images.
-   **Training**: Standard Cross-Entropy loss on FaceForensics++ / Deepfake Dataset.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Created by [Arya Yadav]*
