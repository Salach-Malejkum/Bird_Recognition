import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

start_path_wav = 'C:\\Users\\Margot Poilvet\\Desktop\\BTH\\MTE\\AudioFiles\\Blackbird\\Blackbird('
end_path_wav = ').wav'

start_path_png = 'C:\\Users\\Margot Poilvet\\Desktop\\BTH\\MTE\\Spectrums\\Blackbird\\Blackbird('
end_path_png = ').png'

for i in range(1,18):
    path_wav = start_path_wav + str(i) + end_path_wav
    path_png = start_path_png + str(i) + end_path_png

    # Load the wav file
    signal, sr = librosa.load(path_wav, duration=10)
    
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
    #plt.show()

    plt.savefig(path_png)





