# Team Vocalize

### Overview

Vocalize is a research-oriented assistive communication system investigating whether lightweight deep learning models can reliably recognize frequently used dysarthric utterances under severe data constraints while remaining deployable entirely on mobile hardware.

The project was developed in collaboration with pediatric dysarthric speakers and focuses on constrained-vocabulary communication scenarios where conventional speech-recognition systems often struggle due to atypical articulation patterns and limited training data.

Unlike cloud-based speech systems, Team Vocalize performs inference locally using TensorFlow Lite, enabling low-latency, offline operation.

---

## Motivation

Dysarthria is a motor speech disorder that can significantly reduce speech intelligibility.

Most commercial ASR systems are trained on standard speech distributions and often exhibit degraded performance when applied to dysarthric speech.

This project investigates whether targeted preprocessing, acoustic feature extraction, and efficient neural architectures can improve accessibility for users with atypical speech patterns.

---

## Dataset

A custom dataset was collected from pediatric dysarthric speakers.

Characteristics:

- Custom-recorded speech corpus
- High-frequency communication utterances
- Hindi-dominant vocabulary
- Real-world recording conditions
- Severe dataset scarcity

Target classes:

- BGN (Baseline class, silence)
- Haa
- Hato
- Jao
- Khana
- Nahi
- Ok
- Paani
- Saat
- Saatvik

The constrained vocabulary was intentionally selected to maximize practical communication utility under limited-data conditions.

---

## Signal Processing Pipeline

Raw Audio (16 kHz)
↓
Voice Activity Detection
↓
Energy-Based Alignment
↓
Gain Normalization
↓
Data Augmentation
↓
Mel Spectrogram Generation
↓
CNN Feature Extraction
↓
GRU Temporal Modeling
↓
Classification

---

## Acoustic Feature Extraction

Audio is transformed into Mel-spectrogram representations using:

- Sample Rate: 16 kHz
- FFT Size: 512
- Window Length: 400 samples (25 ms)
- Hop Length: 160 samples (10 ms)
- Mel Bands: 64

### Short-Time Fourier Transform

X(m,k)=Σ x[n]w[n−mH]e^(−j2πkn/N)

### Mel Conversion

m = 2595 log10(1 + f/700)

---

## Model Architecture

The system employs a Convolutional Recurrent Neural Network (CRNN).

### Convolutional Front-End

Conv2D(16)
↓
SepConv2D(32)
↓
SepConv2D(32)
↓
MaxPool
↓
SepConv2D(64)
↓
SepConv2D(64)
↓
MaxPool
↓
SepConv2D(128)
↓
MaxPool

### Temporal Encoder

GRU(64)

### Output

Dense + Softmax

The use of GRUs instead of LSTMs reduces parameter count, memory consumption, and inference latency, making deployment on mobile hardware more practical.

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 16 |
| Epochs | 60 |
| Early Stopping | 8 |
| Seed | 42 |

---

## Mobile Deployment

The trained model is exported to TensorFlow Lite and integrated into an Android application.

Capabilities:

- Offline inference
- Low-latency prediction
- On-device execution
- No cloud dependency

---

## Repository Structure

### Colloquium/

Final research implementation and deployment pipeline.

### Semi-Final-2/

Intermediate experimental baseline used for architecture validation.

### Vocalize App/

Android deployment and inference application.

### outputAI/

Trained models, TensorFlow Lite exports, labels, and evaluation artifacts.

---

## Limitations

This work focuses on constrained-vocabulary communication rather than open-vocabulary speech transcription.

The dataset consists of recordings from a limited number of pediatric dysarthric users and therefore should not be interpreted as a general-purpose dysarthric speech-recognition system.

Future work would include:

- Speaker adaptation
- Larger dysarthric datasets
- Open-vocabulary recognition
- Real-time speech reconstruction
- Personalized language models

---

## Significance

Team Vocalize demonstrates that lightweight spatial–temporal acoustic models can be deployed entirely on mobile hardware to support accessibility-oriented communication tasks under severe data and computational constraints.

The project serves as an exploration of practical machine learning systems for assistive technology, combining digital signal processing, edge computing, and applied deep learning in a real-world setting.

---

## License

This repository is released for educational and research purposes.

Copyright (c) 2026 Team Vocalize

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, and distribute the Software for non-commercial educational and research purposes, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. Any publications, reports, or derivative works based on this project should provide appropriate attribution to the original authors.
3. Commercial use of the Software requires prior written permission from the authors.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Citation

If you use this repository in academic work, please cite:

```text
Team Vocalize: Edge-Based Recognition of Dysarthric Speech
Using Spatial–Temporal Acoustic Modeling.
2026.
```

---

## Acknowledgements

We would like to thank the participating pediatric dysarthric speakers and their families for their time, patience, and willingness to contribute to this research effort. Their participation made this work possible.

We also acknowledge the open-source machine learning, digital signal processing, and mobile development communities whose tools and libraries enabled the development of this project.
```
