### 数据处理：将特征都存到pickle文件中
import os
import librosa
import pickle
from tqdm import tqdm
from math import sqrt
from joblib import Parallel, delayed


noisy_base = '/Share/home/E19301153/IRM_data/noisy_trainset_28spk_wav'
clean_base = '/Share/home/E19301153/IRM_data/clean_trainset_28spk_wav'
n_fft = 320
hop_length = 160

wave_name_list = os.listdir(clean_base) # return type : list 

 
feature_list = []

# for i,wave_name in tqdm(enumerate(wave_name_list)):
def get_feature(wave_name):    
    clean_path = os.path.join(clean_base, wave_name)
    noisy_path = os.path.join(noisy_base, wave_name)

    clean_y, _ = librosa.load(clean_path , sr=16000)
    noisy_y, _ = librosa.load(noisy_path , sr=16000)

    clean_mag, _ = librosa.magphase(librosa.stft(clean_y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    noisy_mag, _ = librosa.magphase(librosa.stft(noisy_y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))

    n_frames = clean_mag.shape[-1]

    feature_list.append([clean_mag, noisy_mag, n_frames])

Parallel(n_jobs=32,backend='threading')(delayed(get_feature)(wave_name) for i , wave_name in tqdm(enumerate(wave_name_list) ))

with open ('./train.pkl','wb') as f:
    pkl.dump(feature_list, f)
    # 格式是 [ [x,y, frames],[x,y,frames]  ]

# # with open('./clean_train.pkl','rb') as f:
del feature_list
