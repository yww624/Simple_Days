### 数据处理：将特征都存到pickle文件中
##
##  涉及的参数：
##  1. noisy_base 
##  2. clean_base
##  3. n_fft:
##  4. hop_length:
##  5. n_jobs:
##  6. sr:
##  使用的时候，添加函数，以及修改一下mode的类型就行了
import os
import librosa
import pickle as pkl
from tqdm import tqdm
from math import sqrt
from joblib import Parallel, delayed

def get_feature(wave_name):    
    clean_path = os.path.join(clean_base, wave_name)
    noisy_path = os.path.join(noisy_base, wave_name)

    clean_y, _ = librosa.load(clean_path , sr=16000)
    noisy_y, _ = librosa.load(noisy_path , sr=16000)

    clean_mag, _ = librosa.magphase(librosa.stft(clean_y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    noisy_mag, _ = librosa.magphase(librosa.stft(noisy_y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))

    n_frames = clean_mag.shape[-1]

    feature_list.append([clean_mag, noisy_mag, n_frames])

def get_file_path(clean_base,noisy_base,wave_name):    
    clean_path = os.path.join(clean_base, wave_name)
    noisy_path = os.path.join(noisy_base, wave_name)
    feature_list.append([clean_mag, noisy_mag, wave_name])

mode = "train"

if mode is "train":
    noisy_base = './noisy_trainset_28spk_wav'
    clean_base = './clean_trainset_28spk_wav'
    pkl_path = './train.pkl'
else:
    noisy_base = './noisy_testset_wav'
    clean_base = './clean_testset_wav'
    pkl_path = './test.pkl'

n_fft = 320
hop_length = 160


wave_name_list = os.listdir(clean_base) # return type : list 
feature_list = []

Parallel(n_jobs=32,backend='threading')(delayed(get_file_path)(wave_name) for i , wave_name in tqdm(enumerate(wave_name_list) ))

with open (pkl_path,'wb') as f:  
    pkl.dump(feature_list, f)

del feature_list
