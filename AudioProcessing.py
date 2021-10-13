import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

file_path = 'C:\\Users\\Margot Poilvet\\Desktop\\BTH\\MTE\\AudioFiles\\'
file_name = 'CyanistesCaeruleus_BlueTit.wav'


# Load the wav file
signal, sr = librosa.load(file_path + file_name, duration=10)

# Plot mel-spectrogram
N_FFT = 1024         
HOP_SIZE = 1024       
N_MELS = 128            
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 1400 

S = librosa.feature.melspectrogram(y=signal,sr=sr,
                                    n_fft=N_FFT,
                                    hop_length=HOP_SIZE, 
                                    n_mels=N_MELS, 
                                    htk=True, 
                                    fmin=FMIN, 
                                    fmax=sr/2) 

plt.figure(figsize=(10, 4))
ax = plt.axes()
ax.set_axis_off()

librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN, y_axis='linear')
#plt.colorbar(format='%+2.0f dB')
plt.show()