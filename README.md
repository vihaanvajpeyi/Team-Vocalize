# Team Vocalize

## Overview

Team Vocalize is an assistive communication project designed to help individuals with dysarthria communicate more effectively using speech recognition technology.

The system recognizes a predefined set of commonly used words and phrases spoken by dysarthric users and converts them into readable text through a mobile application. By leveraging deep learning and on-device inference, the project aims to provide a fast, portable, and accessible communication aid.

---

## Problem Statement

Dysarthria is a motor speech disorder that affects a person's ability to articulate words clearly. Many existing speech recognition systems are trained on typical speech and often perform poorly when presented with dysarthric speech patterns.

Team Vocalize explores whether a machine learning model can be trained specifically on dysarthric speech to recognize frequently used words and commands with higher accuracy.

---

## Features

* Dysarthric speech recognition
* Android mobile application
* TensorFlow Lite deployment
* Real-time audio recording
* Mel-spectrogram based feature extraction
* Deep learning inference directly on device
* Lightweight architecture suitable for mobile hardware

---

## System Architecture

### Training Pipeline

```text
Audio Recordings
        ↓
Preprocessing
        ↓
Mel Spectrogram Generation
        ↓
CNN Feature Extraction
        ↓
GRU Sequence Modeling
        ↓
Softmax Classification
        ↓
Trained Model
```

### Inference Pipeline

```text
User Speech
        ↓
Audio Recording
        ↓
Feature Extraction
        ↓
TensorFlow Lite Model
        ↓
Predicted Word/Phrase
        ↓
Display Output
```

---

## Dataset

The model is trained on a custom dataset consisting of dysarthric speech samples corresponding to a predefined vocabulary.

Example classes include:

* BGN
* Haa
* Hato
* Jao
* Khana
* Nahi
* Ok
* Paani
* Saat
* Saatvik

Each audio sample is converted into a mel spectrogram representation before being used for training.

---

## Machine Learning Model

The project utilizes a hybrid CNN-GRU architecture:

### Convolutional Neural Network (CNN)

Used to extract spatial and frequency-based features from mel spectrograms.

### Gated Recurrent Unit (GRU)

Used to capture temporal dependencies within speech signals.

### Softmax Layer

Outputs the probability distribution across all target classes.

---

## Mobile Deployment

The trained model is converted into TensorFlow Lite format and integrated into the Android application.

Benefits include:

* Low latency inference
* Offline functionality
* Reduced memory usage
* Mobile device compatibility

---

## Technologies Used

### Machine Learning

* Python
* TensorFlow
* TensorFlow Lite
* NumPy
* Librosa

### Mobile Development

* Android Studio
* Java / Kotlin
* TensorFlow Lite Android APIs

### Audio Processing

* Mel Spectrograms
* Audio Normalization
* Feature Extraction

---

## Limitations

Current system capabilities are limited to classification among predefined words and phrases.

The model does not currently support:

* Open-vocabulary speech recognition
* Full sentence transcription
* Continuous conversational speech
* Personalized speaker adaptation

Future work may explore transformer-based speech recognition systems and personalized dysarthric speech adaptation techniques.

---

## Future Improvements

* Larger dysarthric speech dataset
* Expanded vocabulary
* Continuous speech recognition
* Speaker-specific adaptation
* Real-time text-to-speech feedback
* Cloud-assisted inference
* Transformer-based architectures

---

## Team Vocalize

Team Vocalize was developed as an exploration of assistive AI technologies for speech-impaired individuals. The project demonstrates the potential of combining machine learning, mobile computing, and speech processing to improve accessibility and communication.

---

## License

This project is intended for educational and research purposes.
