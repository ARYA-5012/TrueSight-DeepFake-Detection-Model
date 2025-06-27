# 🛡️ TrueSight: A Multimodal Deepfake Detection Framework

> Because fake faces, voices, and videos shouldn't rewrite reality.

![Methodology Flowchart](cha.png)

## 🧠 Overview
**TrueSight** is a unified deepfake detection framework that operates across **image**, **video**, and **audio** modalities. Designed to fight misinformation and protect digital integrity, it combines the strengths of **EfficientNet**, **Lite3D-ResNet**, and **ResNet-1D** fused via a **transformer-based cross-modal attention module**.

The model is hardened through **adversarial training**, optimized with **INT8 quantization**, and benchmarked on **FaceForensics++**, **DFDC**, and **WaveFake**.

---

## 🚀 Key Features

- 🔍 **Multimodal Detection**: Handles image, video, and audio deepfakes
- 🧱 **Transformer-Based Fusion**: Cross-attention for modality synergy
- 🛡️ **Adversarial Robustness**: PGD-based evasion hardening
- ⚡ **Edge Deployment Ready**: Runs under 100ms with INT8 quantization
- 📊 **State-of-the-Art Accuracy**: Beats unimodal baselines on cross-dataset benchmarks

---

## 📁 Project Structure

TrueSight/
├── datasets/
│ ├── images/
│ ├── videos/
│ └── audio/
├── models/
│ ├── efficientnet_b4.py
│ ├── lite3d_resnet.py
│ ├── resnet1d_audio.py
│ └── transformer_fuser.py
├── utils/
│ └── preprocessing.py
├── outputs/
│ └── logs, checkpoints, metrics.json
├── notebooks/
│ └── truesight_pipeline.ipynb
└── main.py

yaml
Copy
Edit

---

## 📦 Installation

```bash
git clone https://github.com/your-username/TrueSight.git
cd TrueSight
pip install -r requirements.txt
⚙️ Dependencies
Python 3.9+

PyTorch 2.0.1

Librosa 0.10.1

Transformers 4.38

Foolbox 3.3.1

OpenCV, NumPy, Matplotlib

🧪 Datasets Used
Modality	Dataset	Type	Samples
Image	FaceForensics++	GAN, Diffusion	150,000
Video	DFDC, Celeb-DF	Swap, Texture	30,000
Audio	WaveFake, ASVspoof	TTS, VC	50,000

🏗️ How to Run
Set up data directories:

bash
Copy
Edit
/TrueSight/datasets/
    ├── images/
    ├── videos/
    └── audio/
Train the models

bash
Copy
Edit
python train.py --modality image
python train.py --modality video
python train.py --modality audio
Fuse and evaluate

bash
Copy
Edit
python evaluate.py --fusion True
Adversarial evaluation

bash
Copy
Edit
python adversarial_test.py --attack pgd
📈 Results Summary
Model	AUC	F1 Score	Inference Time
EfficientNet-B4	92.1%	91.3%	42ms
Lite3D-ResNet	94.8%	93.6%	65ms
ResNet1D (Audio)	88.7%	87.9%	39ms
TrueSight (Fusion)	97.2%	96.3%	94ms

🎯 Use Cases
✅ Social Media Fake Detection APIs

🗞️ Newsroom Verification Pipelines

🎥 Real-time Call & Stream Filtering

🔐 Robustness Add-Ons
PGD Adversarial Noise

Gaussian Blur & Compression (H.264, CRF=30)

Dialect Variation Testing (Planned VCTK integration)

📌 Future Work
🔁 Transfer learning from VCTK (110 accents)

📱 Knowledge distillation for <50MB mobile deployment

🧩 GAN artifact hallucination under CRF > 40

👨‍💻 Authors
Arya Yadav – CSE (AI/ML), Bennett University

Mentored by: Jarvis – AI Assistant, Iron-Clad Logic Mode ⚙️

📜 License
MIT License © 2025 Arya Yadav
