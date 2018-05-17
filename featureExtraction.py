import numpy as np
import os
import file
import librosa.display
import pickle

dir = os.getcwd() + '/data/barking/'

data = list()

for i in range(1,5):
    print('i',i)
    for j in range(1,5):
        print('j',j)
        files = os.listdir(dir+str(i)+'/'+str(j))
        for name in files:
            y, fs = librosa.load(dir+str(i)+'/'+str(j)+'/'+name)
            y = librosa.core.to_mono(y)
            y_16k = librosa.resample(y,fs,16000)
            stft = librosa.core.stft(y_16k,n_fft=2048,hop_length=400,win_length=800,window='hann')
            mel = librosa.feature.melspectrogram(S=np.abs(stft)**2,sr=16000,n_mels=256)
            data.append({'y_16k':y_16k,'emotionType':j,'spec':stft,'melSpec':mel})
            # 데이터 augmentation을 해야할때! 노이즈 넣어줌!
            for k in range(3):
                noise = np.random.normal(0,0.005,size=y_16k.shape)
                y_ = y_16k + noise
                stft = librosa.core.stft(y_,n_fft=2048,hop_length=400,win_length=800,window='hann')
                mel = librosa.feature.melspectrogram(S=np.abs(stft)**2,sr=16000,n_mels=256)
                data.append({'y_16k':y_,'emotionType':j,'spec':stft,'melSpec':mel})


output = open('data_noiseAugmentation.pkl','wb')
pickle.dump(data, output)
output.close()
print('fin')
