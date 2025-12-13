import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

frequency_sampling, audio_signal=wavfile.read("audio_file.wav")

signal = audio_signal[:1000]

if signal.ndim>1:
    signal.data=signal[:, 0]

N=len(signal)

fft_output=np.fft.fft(signal)

magnitude=np.abs(fft_output)

magnitude_half=magnitude[:N//2]

magnitude_half=magnitude_half/N

freq_step=frequency_sampling/N

frequency_axis=np.arrange(0,N//2)*freq_step

plt.figure(figsize=(12,6))
plt.plot(frequency_axis, magnitude_half)
plt.title("Frequency spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()