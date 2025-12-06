import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

frequency_sampling, audio_signal = wavfile.read("audio_file.wav")

print("Sampling frequency:", frequency_sampling)
print("Signal shape:", audio_signal.shape)
print("Signal type:", audio_signal.dtype)
print("Signal duration:", round(audio_signal.shape[0] / float(frequency_sampling), 2), "seconds")

signal = audio_signal[:100]
time_axis = 1000 * np.arange(0, len(signal), 1) / float(frequency_sampling)

plt.plot(time_axis, signal, color='blue')
plt.xlabel("Time (milliseconds)")
plt.ylabel("Amplitude")
plt.title("Input audio signal (first 100 samples)")
plt.show()
