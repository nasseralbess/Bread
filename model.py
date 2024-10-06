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
from scipy.signal import find_peaks

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

def train_autoencoder(catalog_path, data_dir):
    X, y = load_data(catalog_path, data_dir)
    print(f"Loaded {len(X)} samples.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train autoencoder on "normal" data only
    X_train_normal = X_train[y_train == 0]
    
    autoencoder = build_autoencoder((2000, 1))
    history = autoencoder.fit(X_train_normal, X_train_normal, 
                              epochs=1, batch_size=32, 
                              validation_split=0.2, verbose=1)
    
    # Compute reconstruction error
    X_test_pred = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - X_test_pred.reshape(X_test.shape), 2), axis=(1))
    
    # Use reconstruction error as anomaly score
    auc = roc_auc_score(y_test, mse)
    print(f"ROC AUC: {auc:.4f}")
    
    # Determine threshold (e.g., 95th percentile of normal data reconstruction error)
    X_train_normal_pred = autoencoder.predict(X_train_normal)
    mse_normal = np.mean(np.power(X_train_normal - X_train_normal_pred.reshape(X_train_normal.shape), 2), axis=(1))
    threshold = np.percentile(mse_normal, 95)
    
    return autoencoder, threshold, history

  
  
# Set up file paths
catalog_path = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
data_dir = './data/lunar/training/data/S12_GradeA/'

# Train the autoencoder
autoencoder, threshold, history = train_autoencoder(catalog_path, data_dir)

def detect_anomalies(autoencoder, mseed_file, threshold, window_size=2000, stride=1000):
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
    
    anomaly_scores = np.array(anomaly_scores)
    print(sampling_rate)
    print(int(5*sampling_rate))

    # Find peaks in anomaly scores
    peaks, _ = find_peaks(anomaly_scores, height=threshold, distance=int(5*sampling_rate))  # At least 5 seconds apart
    
    # Convert peak indices to timestamps
    event_timestamps = [timestamps[p] for p in peaks]
    
    return event_timestamps, anomaly_scores, timestamps

fp = r'data\lunar\test\data\S15_GradeA\xa.s15.00.mhz.1974-12-15HR00_evid00169.mseed'
event_timestamps, anomaly_scores, timestamps = detect_anomalies(autoencoder, fp, threshold+0.85)
plt.figure(figsize=(15, 5))
ax1 = plt.subplot(2,1,1)
ax1.plot(timestamps, anomaly_scores)
ax1.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
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
plt.show()

print("Detected event timestamps (seconds):")
print(event_timestamps)