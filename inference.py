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



def detect_anomalies(autoencoder, mseed_file, window_size=2000, stride=1000):
    # Read the mseed file
    st = read(mseed_file)
    data = st[0].data
    sampling_rate = st[0].stats.sampling_rate
    
    # Normalize data
    data = (data - np.mean(data)) / np.std(data)
    
    # Sliding window over the data
    timestamps = []
    preds = []
    for i in range(0, len(data) - window_size, stride):
        window = data[i:i+window_size]
        # window = window.reshape(1, window_size)  # Reshape for model input
        preds.append(window)
        # Compute reconstruction error
        # window_pred = autoencoder.predict(window)
        # mse = np.mean(np.power(window - window_pred, 2))
        
        # anomaly_scores.append(mse)
        timestamps.append(i / sampling_rate)  # Convert sample index to time
    preds = np.array(preds)
    scores = autoencoder.predict(preds)
    anomaly_scores = np.mean(np.power(preds - scores.reshape(preds.shape), 2), axis=(1))
    
    # print(sampling_rate)
    # print(int(5*sampling_rate))
    threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
    # Find peaks in anomaly scores
    peaks, _ = find_peaks(anomaly_scores, height=threshold, distance=int(5*sampling_rate))  # At least 5 seconds apart
    
    # Convert peak indices to timestamps
    event_timestamps = [timestamps[p] for p in peaks]
    
    return event_timestamps, anomaly_scores, timestamps, threshold

def inference_viz(fp):
    autoencoder = load_autoencoder()
    event_timestamps, anomaly_scores, timestamps, threshold = detect_anomalies(autoencoder, fp)
    plt.figure(figsize=(15, 8))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(timestamps, anomaly_scores)
    ax1.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    found_anoms = []
    print("Detected event timestamps (seconds):")
    print(event_timestamps)
    for event in event_timestamps:
        #find the index of the event
        event_index = np.where(np.array(timestamps) == event)[0][0]
        found_anoms.append(event_index)

    # check the anomaly scores of the points around the detected events
    event_timestamps = []
    for i in found_anoms:
        median = np.median(np.abs(anomaly_scores[i-3:i+4]))
        mean = np.mean(np.abs(anomaly_scores[i-3:i+4])) 
        print('ratio:,',min(median, mean) / max(median, mean))
        if min(median, mean) / max(median, mean) > 0.7:
            event_timestamps.append(timestamps[i])
    for event in event_timestamps:
        #find the index of the event
        event_index = np.where(np.array(timestamps) == event)[0][0]
        if event_index > 0 and event_index < len(anomaly_scores) - 1:
            anomaly_scores[event_index] = (anomaly_scores[event_index - 1] + anomaly_scores[event_index + 1]) / 2
        
    event_timestamps = sorted(event_timestamps)[:2]
            
    ax1.scatter(event_timestamps, [threshold] * len(event_timestamps), color='red', marker='x', s=100, label='Detected Events')
    ax1.set_title('Anomaly Scores with Detected Events')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Anomaly Score')
    ax1.legend()

    # Plot the seismogram
    st = read(fp)
    tr = st[0]
    tr_times = tr.times()
    tr_data = tr.data
    ax2 = plt.subplot(2,1,2)
    ax2.plot(tr_times,tr_data)
    ax2.set_title('Seismogram')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    for event in event_timestamps:
        ax2.axvline(x=event, color='r',  label='Detected Event')
    plt.show()

    print("Detected event timestamps (seconds):")
    print(event_timestamps)

for directory in os.listdir('./data/lunar/test/data/'):
    for file in os.listdir(f'./data/lunar/test/data/{directory}'):
        print('\n\n\nhere')
        print('file:', f'./data/lunar/test/data/{directory}/{file}')
        if(file.endswith('.mseed')):
            print('file:', f'./data/lunar/test/data/{directory}/{file}')
        # print(f'./data/lunar/test/data/{folder}/{file}')
            inference_viz(f'./data/lunar/test/data/{directory}/{file}')

            
            
    