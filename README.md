# Team Vocalize

### Overview

Team Vocalize investigates whether lightweight deep learning architectures can reliably recognize high-frequency dysarthric utterances under severe data scarcity and deterministic mobile deployment constraints.

The project was motivated by a practical challenge: conventional speech-recognition systems are typically trained on standard speech distributions and often exhibit degraded performance when confronted with dysarthric speech, characterized by atypical articulation patterns, phonetic variability, and reduced speech intelligibility.

Rather than pursuing large-vocabulary speech transcription, Team Vocalize focuses on a constrained communication setting involving frequently used Hindi utterances collected from pediatric dysarthric speakers. The objective is to determine whether a computationally efficient Convolutional Recurrent Neural Network (CRNN) can learn robust acoustic representations suitable for real-time on-device inference.

---

# Dataset

The dataset consists of custom-collected dysarthric speech recordings from pediatric users.

Target utterance classes:

```text
BGN
Haa
Hato
Jao
Khana
Nahi
Ok
Paani
Saat
Saatvik
```

This yields a closed-set classification problem involving ten high-frequency communication tokens.

Unlike benchmark corpora, the dataset captures real-world variability associated with pediatric dysarthric speech production.

---

# Digital Signal Processing Pipeline

## Audio Standardization

All recordings are converted to a fixed representation:

| Parameter        | Value     |
| ---------------- | --------- |
| Sample Rate      | 16,000 Hz |
| Clip Duration    | 1.0 s     |
| Samples per Clip | 16,000    |

```python
SR = 16000
DURATION = 1.0
TARGET_LEN = 16000
```

---

## Energy-Based Voice Activity Centering

To reduce temporal variability, the system performs Voice Activity Detection (VAD) using frame-level energy estimation.

Frame parameters:

| Parameter     | Value  |
| ------------- | ------ |
| Frame Length  | 25 ms  |
| Hop Length    | 10 ms  |
| Active Window | 400 ms |

Frame energy is computed as:

[
E_i = \frac{1}{N}\sum_{n=1}^{N} x_i[n]^2
]

where:

* (x_i[n]) is the (n)-th sample of frame (i)
* (N) is the frame length

The highest-energy contiguous segment is identified and centered within the one-second recording window.

This procedure improves alignment consistency while preserving speaker-specific articulation characteristics.

---

# Mel-Spectrogram Feature Extraction

The system transforms waveforms into two-dimensional time-frequency representations.

## Short-Time Fourier Transform (STFT)

Given a discrete signal (x[n]), the STFT is defined as:

[
X(m,k)=\sum_{n=-\infty}^{\infty}
x[n],w[n-mH],
e^{-j2\pi kn/N}
]

where:

* (w[n]) is the analysis window
* (H) is the hop length
* (N) is the FFT size

---

## Spectrogram Parameters

| Parameter         | Value |
| ----------------- | ----- |
| FFT Size          | 512   |
| Window Length     | 400   |
| Hop Length        | 160   |
| Mel Bands         | 64    |
| Minimum Frequency | 20 Hz |
| Maximum Frequency | 8 kHz |

```python
N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 64
```

---

## Mel Scale Transformation

Human auditory perception is nonlinear with respect to frequency.

The Mel transformation is:

[
m = 2595\log_{10}\left(1+\frac{f}{700}\right)
]

where:

* (f) is frequency in Hertz
* (m) is frequency on the Mel scale

After Mel filtering, power values are converted to decibel space and normalized:

[
S_{dB}=10\log_{10}(S)
]

followed by per-sample z-score normalization.

---

# Neural Network Architecture

The model follows a Convolutional Recurrent Neural Network (CRNN) design.

## Input Representation

```text
64 Mel Bands
×
~101 Time Frames
×
1 Channel
```

---

## Convolutional Front-End

The architecture employs depthwise-separable convolutions to minimize computational complexity while retaining representational capacity.

### Block 0

```text
Conv2D(16)
```

### Block 1

```text
SeparableConv2D(32)
SeparableConv2D(32)
MaxPooling2D
```

### Block 2

```text
SeparableConv2D(64)
SeparableConv2D(64)
MaxPooling2D
```

### Block 3

```text
SeparableConv2D(128)
MaxPooling2D
```

---

## Temporal Modeling Layer

Following convolutional feature extraction:

```text
Permute
↓
Reshape
↓
GRU(64)
```

### GRU Configuration

| Parameter      | Value          |
| -------------- | -------------- |
| Units          | 64             |
| Directionality | Unidirectional |
| Stacked Layers | 1              |

GRUs were selected instead of LSTMs due to:

* Reduced parameter count
* Lower memory footprint
* Fewer gate operations
* Faster inference on mobile hardware

These properties are particularly advantageous under deterministic edge-computing constraints.

---

## Classification Layer

The final dense layer produces class probabilities using Softmax:

[
P(y_i)=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
]

---

# Training Configuration

| Parameter               | Value |
| ----------------------- | ----- |
| Optimizer               | Adam  |
| Learning Rate           | 0.001 |
| Batch Size              | 16    |
| Epochs                  | 60    |
| Early Stopping Patience | 8     |
| Random Seed             | 42    |

---

## Loss Function

The network is trained using sparse categorical cross-entropy.

For a multi-class problem:

[
\mathcal{L}
===========

-\sum_{i=1}^{C}
y_i
\log(\hat y_i)
]

where:

* (C) is the number of classes
* (y_i) is the ground-truth label
* (\hat y_i) is the predicted probability

---

# Data Augmentation

To improve robustness under limited-data conditions, augmentation is applied during training.

| Technique      | Probability |
| -------------- | ----------- |
| Time Shift     | 0.50        |
| Pitch Shift    | 0.40        |
| Time Stretch   | 0.35        |
| Gaussian Noise | 0.50        |

Parameter ranges:

```text
Time Shift: ±0.12 s
Pitch Shift: ±1 semitone
Time Stretch: 0.94× – 1.06×
Noise σ: 0.0006 – 0.007
```

---

# DSP and Training Pipeline

```text
Raw WAV Audio
        │
        ▼
Resampling (16 kHz)
        │
        ▼
Energy-Based VAD
        │
        ▼
Gain Normalization
        │
        ▼
Data Augmentation
        │
        ▼
Mel Spectrogram Generation
        │
        ▼
CNN Feature Extraction
        │
        ▼
GRU Temporal Modeling
        │
        ▼
Softmax Classification
        │
        ▼
Model Training
        │
        ▼
TensorFlow Lite Export
```

---

# Edge Inference Pipeline

```text
Device Microphone
        │
        ▼
Audio Capture
        │
        ▼
Preprocessing
        │
        ▼
Mel Spectrogram Generation
        │
        ▼
TensorFlow Lite Runtime
        │
        ▼
CNN-GRU Inference
        │
        ▼
Softmax Prediction
        │
        ▼
Recognized Utterance
```

---

# Repository Structure

## Colloquium/

Final research and deployment pipeline.

Contains:

* AIModel.py
* TensorFlow Lite export
* Trained models
* Android application
* Evaluation outputs

Represents the final experimental configuration.

---

## Semi-Final-2/

Intermediate experimental baseline.

Contains:

* RecorderApp
* SF2BaseApp
* Earlier preprocessing pipeline
* Earlier mobile deployment framework

Used for iterative architecture validation.

---

## Vocalize App/

Android deployment layer.

Includes:

* Audio capture
* Foreground recording service
* TensorFlow Lite inference
* UI components
* Gradle configuration

Provides fully localized on-device inference.

---

## outputAI/

Model artifacts:

```text
best_model.h5
final_model.h5
2.13-model.tflite
labels.txt
confusion_matrix.png
```

---

# Reproducibility

## Python Environment

```bash
python -m venv vocalize-env

source vocalize-env/bin/activate

pip install tensorflow
pip install librosa
pip install soundfile
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install tqdm
```

---

## Training

```bash
python AIModel.py
```

---

## Android Build

```bash
git clone <repository>

cd Vocalize-App

./gradlew assembleDebug
```

---

## TensorFlow Lite Verification

```bash
adb install app-debug.apk
```

Deploy the generated TFLite model and verify real-time inference through the Android interface.

---

# Research Contributions

This work demonstrates that:

1. Lightweight CRNN architectures can be deployed entirely on-device using TensorFlow Lite.
2. Spatial–temporal acoustic representations extracted from Mel-spectrograms remain informative under dysarthric speech variability.
3. Real-time assistive speech recognition can be achieved without cloud inference.
4. Robust performance is attainable despite severe dataset constraints through targeted preprocessing and augmentation.

Rather than treating dysarthric speech as a generic speech-recognition problem, Team Vocalize investigates how constrained-vocabulary communication systems can be optimized for real-world accessibility applications under practical deployment constraints.
