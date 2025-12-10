#!/usr/bin/env python3
"""
train_vocalize_1s.py

Single-file training pipeline for Vocalize (1-second clips).

Features:
- DURATION = 1.0s, SR = 16000
- Energy-based centering VAD for each clip (centers active region inside 1s)
- On-the-fly augmentation (only training)
- Mel-spectrogram: N_FFT=512, HOP=160, WIN=400, N_MELS=64
- MobileNetV2-like frontend + GRU(64) -> Dense(softmax)
- Per-file train/test split (no leakage)
- Exports model.tflite and labels.txt
"""

import os
import random
import argparse
from pathlib import Path
from glob import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# CONFIG
# -----------------------
SR = 16000
DURATION = 1.0
TARGET_LEN = int(SR * DURATION)

N_FFT = 512
HOP = 160
WIN = 400
N_MELS = 64

RMS_TARGET = 0.02

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Augmentation probabilities (on-the-fly)
AUG_PROB = {
    "time_shift": 0.5,
    "pitch_shift": 0.4,
    "time_stretch": 0.35,
    "noise": 0.5,
}

# Training params (adjustable via CLI)
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 60
DEFAULT_WIDTH_MULT = 0.5
DEFAULT_GRU_UNITS = 64

# -----------------------
# HELPERS
# -----------------------
def list_classes(dataset_root):
    return sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])

def is_wav(fn):
    return fn.lower().endswith(".wav")

def pad_trim(y, target_len=TARGET_LEN):
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def rms(y):
    return np.sqrt(np.mean(y**2))

def apply_gain_normalization(y, target=RMS_TARGET):
    cur = rms(y) + 1e-12
    return y * (target / cur)

# -----------------------
# ENERGY-BASED CENTERING VAD (works for 1s clips)
# -----------------------
def center_active_region(y, sr=SR, active_window_ms=400):
    """
    Find the highest-energy region and center it inside the 1s clip.
    active_window_ms: window length for activity in milliseconds (e.g., 400ms)
    """
    # ensure mono np.array
    y = y.copy().astype(np.float32)
    L = len(y)
    if L != TARGET_LEN:
        y = pad_trim(y, TARGET_LEN)

    # frame-level energy
    frame_len = int(0.025 * sr)  # 25ms
    hop_len = int(0.010 * sr)    # 10ms
    if frame_len < 1:
        frame_len = 1
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len).T  # (n_frames, frame_len)
    energies = (frames ** 2).mean(axis=1)
    # smooth energies with small moving average
    win = max(1, int(0.050 / 0.010))  # 50ms window in frames
    energies_sm = np.convolve(energies, np.ones(win)/win, mode='same')

    # convert active_window_ms to number of frames
    active_frames = max(1, int(round((active_window_ms/1000.0) / (hop_len/sr))))

    # find index of max-energy contiguous block of length active_frames
    if len(energies_sm) <= active_frames:
        # fallback: return original padded clip
        return y
    cumsum = np.cumsum(np.concatenate([[0.0], energies_sm]))
    window_sums = cumsum[active_frames:] - cumsum[:-active_frames]
    idx = np.argmax(window_sums)  # frame index where active block starts
    # convert frame index to sample start
    frame_start_sample = idx * hop_len
    frame_center_sample = frame_start_sample + (active_frames * hop_len) // 2
    # Now center this active center in TARGET_LEN
    center = frame_center_sample
    half = TARGET_LEN // 2
    start = int(center - half)
    # build new array by centering. since original length = TARGET_LEN, we will shift accordingly
    out = np.zeros_like(y)
    # compute overlap
    if start >= 0:
        # copy y[0: TARGET_LEN - start] into out[start:]
        length = min(TARGET_LEN - start, TARGET_LEN)
        out[start:start+length] = y[0:length]
    else:
        # negative start: copy from y[-start: ...] to out[0:...]
        src_start = -start
        length = min(TARGET_LEN - src_start, TARGET_LEN)
        out[0:length] = y[src_start:src_start+length]
    # If centering failed (all zeros), return original
    if out.sum() == 0:
        return y
    return out

# -----------------------
# Augmentation (on-the-fly)
# -----------------------
def augment_waveform(y, sr=SR):
    y = y.copy()
    # small random time shift
    if random.random() < AUG_PROB["time_shift"]:
        shift = int(random.uniform(-0.12, 0.12) * sr)
        y = np.roll(y, shift)
        if shift > 0:
            y[:shift] = 0.0
        else:
            y[shift:] = 0.0

    # pitch shift
    if random.random() < AUG_PROB["pitch_shift"]:
        steps = random.uniform(-1.0, 1.0)
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        except Exception:
            pass

    # time stretch via mild resample
    if random.random() < AUG_PROB["time_stretch"]:
        rate = random.uniform(0.94, 1.06)
        try:
            tmp_sr = int(sr * rate)
            stretched = librosa.resample(y=y, orig_sr=sr, target_sr=tmp_sr)
            if tmp_sr != sr:
                stretched = librosa.resample(y=stretched, orig_sr=tmp_sr, target_sr=sr)
            y = pad_trim(stretched, TARGET_LEN)
        except Exception:
            pass

    # add small gaussian noise
    if random.random() < AUG_PROB["noise"]:
        noise = np.random.normal(0, random.uniform(0.0006, 0.007), len(y))
        y = y + noise

    # clip final
    y = np.clip(y, -1.0, 1.0)
    return y

# -----------------------
# Mel feature
# -----------------------
def waveform_to_melspec(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, win_length=WIN):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # per-sample normalization
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)

# -----------------------
# Keras Sequence for generator
# -----------------------
from tensorflow.keras.utils import Sequence

class AudioSequence(Sequence):
    def __init__(self, items, label_encoder, batch_size=DEFAULT_BATCH, shuffle=True, augment=False, vad_active_ms=400):
        """
        items: list of (path, class_name)
        label_encoder: fitted LabelEncoder
        augment: boolean (only True for training)
        vad_active_ms: window used by VAD centering
        """
        self.items = list(items)
        self.le = label_encoder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.vad_active_ms = vad_active_ms
        self.indexes = np.arange(len(self.items))
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.items) / float(self.batch_size))))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.items))
        batch = [self.items[i] for i in range(start, end)]

        X = []
        y = []
        for path, label in batch:
            try:
                wav, sr = sf.read(path)
            except Exception:
                wav, sr = librosa.load(path, sr=SR)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            # if sample rate mismatch
            if sr != SR:
                try:
                    wav = librosa.resample(y=wav, orig_sr=sr, target_sr=SR)
                    sr = SR
                except Exception:
                    wav, sr = librosa.load(path, sr=SR)
            # pad/trim to target len
            wav = pad_trim(wav, TARGET_LEN)
            # Center active region (VAD)
            wav = center_active_region(wav, sr=SR, active_window_ms=self.vad_active_ms)
            # normalize amplitude first (preserve relative differences)
            wav = apply_gain_normalization(wav, target=RMS_TARGET)
            # augmentation only for training
            if self.augment:
                wav = augment_waveform(wav, sr=SR)
                # do not forcibly renormalize to keep some realistic variation
                # but clip to [-1,1]
                wav = np.clip(wav, -1.0, 1.0)

            mel = waveform_to_melspec(wav, sr=SR)
            X.append(mel[..., np.newaxis])
            y.append(label)
        X = np.stack(X, axis=0)
        y = np.array(self.le.transform(y), dtype=np.int32)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.items = [self.items[i] for i in self.indexes]

# -----------------------
# Model: MobileNetV2-like frontend + GRU
# -----------------------
def conv_block(x, filters, kernel=3, strides=1):
    x = layers.SeparableConv2D(filters, kernel, padding="same", strides=strides, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_model(input_shape, n_classes, width_multiplier=DEFAULT_WIDTH_MULT, gru_units=DEFAULT_GRU_UNITS, dropout=0.2):
    inp = layers.Input(shape=input_shape, name="input_mel")  # (n_mels, time, 1)
    x = inp
    base_filters = max(8, int(32 * width_multiplier))

    # initial conv
    x = layers.Conv2D(base_filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # lightweight separable conv blocks
    x = conv_block(x, int(64 * width_multiplier))
    x = conv_block(x, int(64 * width_multiplier))
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, int(128 * width_multiplier))
    x = conv_block(x, int(128 * width_multiplier))
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, int(256 * width_multiplier))
    x = layers.MaxPool2D(2)(x)

    # permute to (time, freq, channels)
    shape_after = K.int_shape(x)  # (None, freq, time, channels)
    freq_dim = shape_after[1]
    time_dim = shape_after[2]
    channels = shape_after[3]
    # safe permute
    x = layers.Permute((2,1,3))(x)  # (None, time, freq, channels)

    # collapse freq and channels to features per time step
    # recalc shape
    shape_after = K.int_shape(x)
    time_dim = shape_after[1]
    freq_dim = shape_after[2]
    channels = shape_after[3]
    feat_dim = freq_dim * channels
    x = layers.Reshape((time_dim, feat_dim))(x)  # (time, features)

    # GRU
    x = layers.GRU(gru_units, return_sequences=False)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out, name="mobilenetv2_gru_1s")
    return model

# -----------------------
# Utility: compute features for a list (no augmentation)
# -----------------------
def compute_features_for_list(items, le, vad_active_ms=400):
    X = []
    y = []
    for path, label in items:
        try:
            wav, sr = sf.read(path)
        except Exception:
            wav, sr = librosa.load(path, sr=SR)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            try:
                wav = librosa.resample(y=wav, orig_sr=sr, target_sr=SR)
            except Exception:
                wav, sr = librosa.load(path, sr=SR)
        wav = pad_trim(wav, TARGET_LEN)
        wav = center_active_region(wav, sr=SR, active_window_ms=vad_active_ms)
        wav = apply_gain_normalization(wav, target=RMS_TARGET)
        mel = waveform_to_melspec(wav, sr=SR)
        X.append(mel[..., np.newaxis])
        y.append(label)
    X = np.stack(X, axis=0)
    y = np.array(le.transform(y), dtype=np.int32)
    return X, y

# -----------------------
# Training pipeline
# -----------------------
def train_pipeline(dataset_root, out_dir, batch_size=DEFAULT_BATCH, epochs=DEFAULT_EPOCHS,
                   width_multiplier=DEFAULT_WIDTH_MULT, gru_units=DEFAULT_GRU_UNITS,
                   val_split=0.2, test_size=0.0, vad_active_ms=400):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = list_classes(dataset_root)
    if not classes:
        raise SystemExit("No class folders found under dataset root.")
    print("Detected classes:", classes)

    # Collect files (case-insensitive)
    items = []
    for cl in classes:
        folder = os.path.join(dataset_root, cl)
        for fn in os.listdir(folder):
            if is_wav(fn):
                items.append((os.path.join(folder, fn), cl))
    print(f"Total WAV files found: {len(items)}")

    # Label encode
    le = LabelEncoder()
    le.fit([cl for _, cl in items])
    labels = list(le.classes_)
    print("Label order:", labels)

    # Split by file-level stratified split
    file_paths = [p for p, c in items]
    file_labels = [c for p, c in items]
    X_train_files, X_val_files, y_train_files, y_val_files = train_test_split(
        file_paths, file_labels, test_size=val_split, random_state=RANDOM_SEED, stratify=file_labels
    )
    train_items = list(zip(X_train_files, y_train_files))
    val_items = list(zip(X_val_files, y_val_files))

    print(f"Train files: {len(train_items)}, Val files: {len(val_items)}")

    # Generators
    train_gen = AudioSequence(train_items, le, batch_size=batch_size, shuffle=True, augment=True, vad_active_ms=vad_active_ms)
    val_gen = AudioSequence(val_items, le, batch_size=batch_size, shuffle=False, augment=False, vad_active_ms=vad_active_ms)

    # Build a sample to get input shape
    sample_X, _ = val_gen[0]
    input_shape = sample_X.shape[1:]  # (n_mels, time, 1)
    print("Input shape:", input_shape)

    model = build_model(input_shape, n_classes=len(labels), width_multiplier=width_multiplier, gru_units=gru_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Callbacks
    best_h5 = out_dir / "best_model.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(best_h5), save_best_only=True, monitor="val_loss")
    ]

    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)

    # Fit
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=callbacks,
                        workers=2,
                        use_multiprocessing=False)
    
    # IF CODE HAS AN ERROR, DELETE THIS PART -----
    
    # After model.fit(...)
    print("\n==== FINAL TRAINING METRICS ====")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"Training Accuracy: {train_acc:.4f}, Training Loss: {train_loss:.4f}")

    print("\n==== FINAL VALIDATION METRICS ====")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

    # ------------ TO HERE

    # Evaluate - compute features for val set explicitly to get confusion matrix
    X_val, y_val = compute_features_for_list(val_items, le, vad_active_ms=vad_active_ms)
    preds = np.argmax(model.predict(X_val, batch_size=batch_size), axis=1)
    print("Validation classification report:")
    print(classification_report(y_val, preds, target_names=labels))

    # Confusion matrix plot
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix to {out_dir / 'confusion_matrix.png'}")

    # Save labels.txt
    labels_txt = out_dir / "labels.txt"
    with open(labels_txt, "w") as f:
        f.write("\n".join(labels))
    print(f"Saved labels to {labels_txt}")

    # Save final keras model (best weights already restored)
    final_h5 = out_dir / "final_model.h5"
    model.save(final_h5)
    print(f"Saved Keras model to {final_h5}")

    # TFLite export - try pure BUILTINS first, then fall back
    tflite_path = out_dir / "model.tflite"
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Exported TFLite model to {tflite_path} (BUILTINS)")
    except Exception as e:
        print("Pure BUILTINS TFLite export failed:", e)
        print("Attempting fallback with SELECT_TF_OPS (may require TensorFlow >= appropriate version on device).")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"Exported TFLite model to {tflite_path} with SELECT_TF_OPS")
        except Exception as e2:
            print("TFLite export failed entirely:", e2)
            print("You may need to check TF version / ops compatibility. Keras model saved at:", final_h5)

    print("Done. Artifacts in:", out_dir)
    return out_dir

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset root (dataset/<class>/*.wav)")
    p.add_argument("--out", required=True, help="Output directory to save models and labels")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--width_multiplier", type=float, default=DEFAULT_WIDTH_MULT)
    p.add_argument("--gru_units", type=int, default=DEFAULT_GRU_UNITS)
    p.add_argument("--val_split", type=float, default=0.20)
    p.add_argument("--vad_ms", type=int, default=400, help="Active window (ms) used by VAD centering")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args.dataset, args.out, batch_size=args.batch, epochs=args.epochs,
                    width_multiplier=args.width_multiplier, gru_units=args.gru_units,
                    val_split=args.val_split, vad_active_ms=args.vad_ms)
