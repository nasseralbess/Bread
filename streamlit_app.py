import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from obspy import read
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, stft
import streamlit as st

def load_data(catalog_path, data_dir):
    catalog = pd.read_csv(catalog_path)
    
    X = []
    y = []
    
    for _, row in catalog.iterrows():
        filename = row['filename']
        mseed_files = glob.glob(os.path.join(data_dir, f"{filename}*.mseed"))
        
        if not mseed_files:
            print(f"Warning: No mseed file found for {filename}")
            continue
        
        mseed_file = mseed_files[0]
        
        try:
            st = read(mseed_file)
            data = st[0].data
            data = (data - np.mean(data)) / np.std(data)
            
            # Extract windows from the entire signal
            window_size = 2000
            stride = 1000
            for i in range(0, len(data) - window_size, stride):
                window = data[i:i+window_size]
                X.append(window)
                
                # Check if this window contains the event
                event_time = int(float(row['time_rel(sec)']) * st[0].stats.sampling_rate)
                y.append(1 if i <= event_time < i+window_size else 0)
            
        except Exception as e:
            print(f"Error processing file {mseed_file}: {str(e)}")
    
    if not X:
        raise ValueError("No valid data was loaded. Please check your file paths and data.")
    
    return np.array(X), np.array(y)

def build_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, 5, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2)(x)
    
    # Decoder
    x = layers.Conv1D(8, 5, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 5, activation='linear', padding='same')(x)
    
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def load_autoencoder():
    autoencoder = build_autoencoder((2000, 1))
    autoencoder.load_weights('./models/autoencoder_1_epoch.h5')
    return autoencoder



def detect_anomalies(autoencoder, data, sampling_rate, window_size=2000, stride=1000):
    # Normalize data
    data = (data - np.mean(data)) / np.std(data)
    
    # Sliding window over the data
    timestamps = []
    preds = []
    for i in range(0, len(data) - window_size, stride):
        window = data[i:i+window_size]
        preds.append(window)
        timestamps.append(i / sampling_rate)  # Convert sample index to time
    preds = np.array(preds)
    scores = autoencoder.predict(preds)
    anomaly_scores = np.mean(np.power(preds - scores.reshape(preds.shape), 2), axis=(1))
    
    threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
    # Find peaks in anomaly scores
    peaks, _ = find_peaks(anomaly_scores, height=threshold, distance=int(5*sampling_rate))  # At least 5 seconds apart
    
    # Convert peak indices to timestamps
    event_timestamps = [timestamps[p] for p in peaks]
    
    return event_timestamps, anomaly_scores, timestamps, threshold

def main():
    st.title("Seismic Anomaly Detection")

    uploaded_file = st.file_uploader("Choose an MSEED file", type="mseed")
    
    if uploaded_file is not None:
        # Read the MSEED file
        st.write("Processing file...")
        stream = read(uploaded_file)
        trace = stream[0]
        data = trace.data
        sampling_rate = trace.stats.sampling_rate

        # Load the autoencoder
        autoencoder = load_autoencoder()

        # Detect anomalies
        event_timestamps, anomaly_scores, timestamps, threshold = detect_anomalies(autoencoder, data, sampling_rate)

        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot anomaly scores
        ax1.plot(timestamps, anomaly_scores)
        ax1.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax1.scatter(event_timestamps, [threshold] * len(event_timestamps), color='red', marker='x', s=100, label='Detected Events')
        ax1.set_title('Anomaly Scores with Detected Events')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Anomaly Score')
        ax1.legend()

        # Plot the seismogram
        ax2.plot(trace.times(), trace.data)
        ax2.set_title('Seismogram')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        for event in event_timestamps:
            ax2.axvline(x=event, color='r',  label='Detected Event')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Detected event timestamps (seconds):")
        st.write(event_timestamps)

if __name__ == "__main__":
    main()