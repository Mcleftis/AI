import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

Fs=44100
f0=440
T=3
Amplitude=1

N=Fs*T
t=np.linspace(0, T, N, endpoint=False)

x=Amplitude*np.sin(2*np.pi*f0*t)

fft_output=np.fft.fft(x)

magnitude=np.abs(fft_output)[:N//2]

freqs=np.fft.fftfreq(N, 1/Fs)[:N//2]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:1000], x[:1000])
plt.title('Pedio xronou{fo(Hz)}')
plt.xlabel('Xronos(second)')
plt.ylabel('Platos')
plt.grid(True)

plt.subplot(2, 1, 2)


plt.plot(freqs, 2 * magnitude / N) 

plt.title('Fasma')
plt.xlabel('Frequency (Hz)') 
plt.ylabel("Normalized Magnitude")
plt.xlim(0, 1000) 
plt.grid(True)
plt.tight_layout() 
plt.show()