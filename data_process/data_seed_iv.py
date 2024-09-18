import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from scipy.signal import welch, butter, lfilter
from pykalman import KalmanFilter
import math
import re


# Deap description里没有提到降噪，只说了去artifacts，但其实带通滤波和ICA去artifacts已经包含一部分降噪功能了

# coeffs = pywt.wavedec(eeg_signal, 'db4', level=4) 可以用小波分解再平滑一些，或者EM平滑/LDS平滑


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b, a] = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def filter_data(data, fs=200):
    print('data shape:', data.shape)
    chan, time = data.shape
    bands = 5
    filtered = np.zeros((chan, bands, time))
    filtered[:, 0, :] = butter_bandpass_filter(data, 1, 4, fs)
    filtered[:, 1, :] = butter_bandpass_filter(data, 4, 8, fs)
    filtered[:, 2, :] = butter_bandpass_filter(data, 8, 14, fs)
    filtered[:, 3, :] = butter_bandpass_filter(data, 14, 31, fs)
    filtered[:, 4, :] = butter_bandpass_filter(data, 31, 45, fs)
    print('filter shape:', filtered.shape)
    return filtered


def LDS_filter(all_data):
    # all data shape: trial, channel, time, band
    channel, band, time = all_data.shape
    filtered = np.zeros(all_data.shape)
    for chan in range(channel):
        for b in range(band):
            print("chan, b:", chan, b)
            kf = KalmanFilter(initial_state_mean=0, observation_covariance=1)
            (filtered_c, _) = kf.smooth(all_data[chan, b, :])
            filtered[chan, b, :] = filtered_c[:, 0]
            print("filtered_c:", filtered_c.shape)
    return filtered


def load_data_label(file_path, how='valence'):  # , how='valence'
    dat = loadmat(file_path)
    raw_data = dat['data'][:, 0:32, 384:]  # 60s data; 40 32 3*128=384
    label = dat['labels'][:, 0:2]  # get valence and arousal

    if how == 'valence':
        # new_label = label[:, 0]  # if regression
        new_label = np.where(label[:, 0] > 5, 1, 0)
    elif how == 'arousal':
        # new_label = label[:, 1]
        new_label = np.where(label[:, 1] > 5, 1, 0)
    elif how == 'all':
        new_label = np.zeros(label.shape[0], dtype=int)
        new_label[(label[:, 0] > 5) & (label[:, 1] < 5)] = 1
        new_label[(label[:, 0] < 5) & (label[:, 1] > 5)] = 2
        new_label[(label[:, 0] > 5) & (label[:, 1] > 5)] = 3
    else:
        raise ValueError

    print(file_path, new_label)

    return raw_data, new_label


def splitTrainTest(data, labels, save_folder):
    # [B, Channels, Bands, Time]
    print("x shape: ", data.shape, ", y shape: ", labels.shape)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    for cat in ['train', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # local() 是当前def中的所有变量构成的字典
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(save_folder, f"{cat}.npz"), x=_x, y=_y)


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance + 1e-6) / 2


def compute_featuresDE(filtered_data, baseline, num_windows):  # num_windows: 每s要分成多少个windows
    data = filtered_data[:, :, 3 * 128:, :].transpose((0, 2, 1, 3))  # 40 7680 32 4
    videos, t, chans, bands = data.shape
    features = np.zeros((videos, 60 * num_windows, chans, bands))
    for video in range(videos):
        for chan in range(chans):
            for band in range(bands):
                trial = data[video, :, chan, band]
                step_size = 14  # 差不多是109ms
                window_size = len(trial) - (60 * num_windows - 1) * step_size  #
                # print('overlap:', window_size-step_size, '; window_size:', window_size)
                for i in range(60 * num_windows):
                    DEfeature = compute_DE(trial[i * step_size:i * step_size + window_size])
                    normed = DEfeature - baseline[video, chan, band]
                    features[video, i, chan, band] = normed
    return features


if __name__ == '__main__':
    raw_path = '/data/Anaiis/Data/SEED_IV/eeg_raw_data/1'
    feats_path = '/data/Anaiis/Data/SEED_IV/eeg_feature_smooth/1'
    window_len = 800

    for matfile in os.listdir(raw_path):
        data = loadmat(os.path.join(raw_path, matfile))
        de_data = loadmat(os.path.join(feats_path, matfile))
        all_trial = []
        all_label = []
        all_de = []
        label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
        smooth = False
        save_folder = f'/data/Anaiis//Data/SEED_IV/len_{window_len}/{matfile}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for trial in range(24):
            pattern = re.compile(rf'.*_eeg{trial+1}$')
            eeg_key = [key for key in data.keys() if pattern.match(key)]
            tmp_trial = data[eeg_key[0]]
            tmp_de_feat = de_data['de_LDS' + str(trial+1)]
            # reshape de
            tmp_de_feat = tmp_de_feat.transpose((1, 0, 2))  # sample, 62, 5
            # 带通滤波
            filtered = filter_data(tmp_trial)  # 62, 5, time
            if smooth:
                filtered = LDS_filter(filtered)
            # 划分样本
            trial_len = len(tmp_trial[1])
            num_samples = trial_len // window_len
            split_filter = np.reshape(filtered[:, :, :num_samples * window_len], (62, 5, num_samples, window_len))
            # 62, 5, samples, window_len
            split_filter = split_filter.transpose((2, 0, 1, 3))  # (sample, 62, 5, 20)
            # all_label.append([label[trial]] * num_samples)
            if trial == 0:
                all_trial = split_filter
                all_label = [label[trial]] * num_samples
                all_de = tmp_de_feat
            else:
                all_trial = np.concatenate([all_trial, split_filter], axis=0)
                all_label = np.concatenate([all_label, [label[trial]] * num_samples])
                all_de = np.concatenate([all_de, tmp_de_feat], axis=0)
        print(f'{matfile}:', 'all_trial:', all_trial.shape, 'all_label:', all_label.shape, np.unique(all_label))
        print('de:', all_de.shape)
        np.save(os.path.join(save_folder, "data.npy"), all_trial)
        np.save(os.path.join(save_folder, "label.npy"), all_label)
        np.save(os.path.join(save_folder, "de.npy"), all_de)
