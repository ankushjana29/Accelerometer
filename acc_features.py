import pandas as pd
import numpy as np
from scipy import signal
import librosa

# Load accelerometer data from CSV file
df = pd.read_csv("C:/Users/admin/Desktop/internship/accelerometer/accelerometer_data.csv")
data = df.iloc[:, 1:].to_numpy()

# Time-domain features
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
rms = np.sqrt(np.mean(data**2, axis=0))
zcr = np.mean(np.abs(np.diff(np.sign(data))), axis=0)
ssc = np.sum(np.abs(np.diff(np.sign(np.diff(data)))), axis=0)

# Frequency-domain features
f, psd = signal.welch(data, axis=0)
df = f[1] - f[0]
dom_freq = np.abs(f[np.argmax(np.abs(psd), axis=0)])
spec_entropy = -np.sum(psd * np.log2(psd + 1e-12), axis=0)

# Time-frequency domain features
f, t, spectrogram = signal.spectrogram(data, fs=100.0, nperseg=100)  #"fs"=sampling frequency and "nperseg"= number of samples per segment
mfcc = librosa.feature.mfcc(y=data, sr=100.0, n_mfcc=13)
mfcc_mean = np.mean(np.abs(mfcc), axis=1)

# Print the extracted features
print("Mean: ", mean)
print("Standard deviation: ", std)
print("RMS: ", rms)
print("ZCR: ", zcr)
print("SSC: ", ssc)
print("Dominant frequency: ", dom_freq)
print("Spectral entropy: ", spec_entropy)
print("MFCC mean: ", mfcc_mean)

