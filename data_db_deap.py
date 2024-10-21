import numpy as np
import os
import math
from scipy.io import loadmat
from pykalman import KalmanFilter
from scipy.signal import welch, butter, lfilter
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b, a] = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def filter_data(data, fs=128):
    video, chan, time = data.shape
    bands = 4
    filtered = np.zeros((video, chan, time, bands))
    filtered[:, :, :, 0] = butter_bandpass_filter(data, 4, 8, fs)
    filtered[:, :, :, 1] = butter_bandpass_filter(data, 8, 14, fs)
    filtered[:, :, :, 2] = butter_bandpass_filter(data, 14, 31, fs)
    filtered[:, :, :, 3] = butter_bandpass_filter(data, 31, 45, fs)
    return filtered


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance + 1e-6) / 2


def load_data_label(file_path):  # , how='valence'
    dat = loadmat(file_path)
    raw_data = dat['data'][:, 0:32, :]  # 63s data; 40 32 63*128=8064
    label = dat['labels'][:, 0:2]  # get valence and arousal

    # if how == 'valence':
    #     # new_label = label[:, 0]  # if regression
    #     new_label = np.where(label[:, 0] > 5, 1, 0)
    # elif how == 'arousal':
    #     # new_label = label[:, 1]
    #     new_label = np.where(label[:, 1] > 5, 1, 0)
    # elif how == 'all':
    new_label = np.zeros(label.shape[0], dtype=int)
    new_label[(label[:, 0] > 5) & (label[:, 1] < 5)] = 1
    new_label[(label[:, 0] < 5) & (label[:, 1] > 5)] = 2
    new_label[(label[:, 0] > 5) & (label[:, 1] > 5)] = 3
    # else:
    #     raise ValueError

    return raw_data, new_label


def load_data_label_regression(file_path):
    dat = loadmat(file_path)
    raw_data = dat['data'][:, 0:32, :]  # 63s data; 40 32 63*128=8064
    label = dat['labels'][:, 0:2]  # get valence and arousal
    all_labels = []
    for i in range(label.shape[0]):
        if 1 <= label[i, 0] <= 4.8 and 1 <= label[i, 1] <= 4.8:
            all_labels.append(1)
        elif 5.2 < label[i, 0] <= 9 and 1 <= label[i, 1] <= 4.8:
            all_labels.append(2)
        elif 1 <= label[i, 0] <= 4.8 and 5.2 < label[i, 1] <= 9:
            all_labels.append(3)
        elif 5.2 < label[i, 0] <= 9 and 5.2 < label[i, 1] <= 9:
            all_labels.append(4)
        else:
            all_labels.append(0)
    non_zero_indices = []

    # 使用enumerate遍历列表元素及其索引
    for index, value in enumerate(all_labels):
        if value != 0:
            non_zero_indices.append(index)

    removed_data = raw_data[non_zero_indices, :, :]
    removed_label = label[non_zero_indices, :]

    return removed_data, removed_label


def compute_baseline(filtered_data):
    white = filtered_data[:, :, :3 * 128, :]
    baseline = np.zeros((white.shape[0], white.shape[1], white.shape[3]))
    for i in range(white.shape[0]):
        for j in range(white.shape[1]):
            for k in range(white.shape[3]):
                _, baseline_3s = welch(white[i, j, :, k], fs=128, nperseg=128)
                baseline[i, j, k] = np.sum(baseline_3s)
    return baseline  # video, chan, band


def compute_baselineDE(filtered_data):
    white = filtered_data[:, :, -3 * 128:, :]
    baseline = np.zeros((white.shape[0], white.shape[1], white.shape[3]))
    for i in range(white.shape[0]):
        for j in range(white.shape[1]):
            for k in range(white.shape[3]):
                baseline_3s = compute_DE(white[i, j, :, k])
                baseline[i, j, k] = baseline_3s
    return baseline  # video, chan, band


def compute_features(filtered_data, baseline, num_windows):  # window_len unit:(Hz)
    data = filtered_data[:, :, 3 * 128:, :].transpose((0, 2, 1, 3))  # 40 7680 32 4
    videos, t, chans, bands = data.shape
    features = np.zeros((videos, 60 * num_windows, chans, bands))
    for video in range(videos):
        for chan in range(chans):
            for band in range(bands):
                trial = data[video, :, chan, band]
                step_size = len(trial) // num_windows
                window_size = len(trial) - (num_windows - 1) * step_size
                print('overlap:', window_size - step_size, '; window_size:', window_size)
                for step in range(0, len(trial) - window_size + 1, step_size):
                    _, psds = welch(trial[step:step + window_size], fs=128, nperseg=128)
                    psd = np.sum(psds)
                    normed = psd / baseline[video, chan, band]
                    features[video, step, chan, band] = 10 * np.log(normed + 1e-6)
    return features


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


def compute_features_96(filtered_data, baseline, window_len):  # , baseline  num_windows: 每s要分成多少个windows
    data = filtered_data[:, :, :-3 * 128, :].transpose((0, 2, 1, 3))  # 40 7680 32 4
    videos, t, chans, bands = data.shape
    # data = data.reshape((-1, chans, bands))
    # data = data.reshape((videos * 30, 256, 32, 4))
    features = np.zeros((videos, t // window_len, chans, bands))
    # step_size = 20
    # window_size = 96
    for v in range(videos):
        for chan in range(chans):
            for band in range(bands):
                for i in range(t // window_len):
                    trial = data[v, i*window_len: i*window_len+window_len, chan, band]
                    DEfeature = compute_DE(trial)
                    normed = DEfeature - baseline[v, chan, band]
                    features[v, i, chan, band] = normed
    return features


def filterLDS(features):  # num_windows: 每s要分成多少个windows
    seg, i, chan, band = features.shape
    # feat_continue = np.reshape(features, (-1, chan, band))  # seg, i, chan, band
    tmp = np.zeros(features.shape)
    for sample in range(seg):
        x = np.linspace(0, 1, i)
        for sing_chan in range(chan):
            for sing_band in range(band):
                signal = features[sample, :, sing_chan, sing_band]
                smoothed_signal = lowess(signal, x, frac=0.8)
                tmp[sample, :, sing_chan, sing_band] = smoothed_signal[:, 1]
    return tmp


def seperate(features, T, init_label):
    # T 代表想要每个cube里包含多少个时间切片
    videos, steps, chans, bands = features.shape
    non_seperate = features.reshape((videos * steps, chans, bands))
    seperated = non_seperate.reshape((int(videos * steps / T), T, chans, bands))
    labels = np.repeat(init_label, steps / T, axis=0)
    print(seperated.shape, labels.shape)
    return seperated, labels


# 应该是连续的


def generate_train_val_test(seperated, labels, save_folder):
    # [B, T, N ,C]

    print("x shape: ", seperated.shape, ", y shape: ", labels.shape)

    num_samples = seperated.shape[0]
    num_test = round(num_samples * 0.1)
    num_train = round(num_samples * 0.8)
    num_val = num_samples - num_train - num_test

    # 训练集
    x_train, y_train = seperated[:num_train], labels[:num_train]

    # 验证集
    x_val, y_val = seperated[num_train:num_train + num_val], labels[num_train:num_train + num_val]

    # 测试集
    x_test, y_test = seperated[num_train + num_val:], labels[num_train + num_val:]

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # local() 是当前def中的所有变量构成的字典

        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            # 保存多个数组，按照你定义的key字典保存，compressed表示它是一个压缩文件
            os.path.join(save_folder, f"{cat}.npz"),
            x=_x,
            y=_y)


def generate_train_val_test_shuffle(seperated, labels, save_folder):
    # [B, T, N ,C]

    print("x shape: ", seperated.shape, ", y shape: ", labels.shape)

    x_train, x_temp, y_train, y_temp = train_test_split(seperated, labels, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # local() 是当前def中的所有变量构成的字典

        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            # 保存多个数组，按照你定义的key字典保存，compressed表示它是一个压缩文件
            os.path.join(save_folder, f"{cat}.npz"),
            x=_x,
            y=_y)


if __name__ == '__main__':
    # for filename in os.listdir('../GELM/data_preprocessed_matlab/'):
    # for filename in os.listdir('./data/data_preprocessed_matlab/'):
    # dta = np.load('../Effect_Att/Data/len_96/False/s01/de.npy')
    # abel = np.load('../Effect_Att/Data/len_96/False/s01/label.npy')
    # print(dta.shape, abel.shape)
    file_path = f'/data/Anaiis/Data/DEAP_original/'
    for filename in os.listdir(file_path):
        for window_len in [96]:
            n = int(7680 / window_len)
            raw_data, new_label = load_data_label(os.path.join(file_path, filename))
            labels = np.repeat(new_label, n, axis=0)
            filtered = filter_data(raw_data)  # (40, 32, 8064, 4)
            data = filtered[:, :, :-3 * 128, :].transpose((0, 1, 3, 2))
            print("shape1", data.shape) #(40, 32, 4, 7680)
            data = data.reshape((40, 32, 4, window_len, -1))
            print(data.shape)
            data = data.transpose((0, 4, 1, 2, 3))
            print("shape2", data.shape)  # 40，n, 32, 4, window_len
            data = data.reshape((-1, 32, 4, window_len))
            print(data.shape)
            baseline = compute_baselineDE(filtered)
            # window_num = 9  # 需被T整除
            features = compute_features_96(filtered, baseline, window_len)  # (40, 80, 32, 4)
            features_LDS = filterLDS(features)
            print(features_LDS.shape)
            feat = features_LDS.reshape((-1, 32, 4))
            if not os.path.exists(f'/data/Anaiis/Data/DEAP/len_{window_len}/{filename[:3]}'):
                os.makedirs(f'/data/Anaiis/Data/DEAP/len_{window_len}/{filename[:3]}')
            np.save(f'/data/Anaiis/Data/DEAP/len_{window_len}/{filename[:3]}/de_LDS.npy', feat)
            np.save(f'/data/Anaiis/Data/DEAP/len_{window_len}/{filename[:3]}/data.npy', data)
            np.save(f'/data/Anaiis/Data/DEAP/len_{window_len}/{filename[:3]}/label.npy', labels) 
    # features_LDS = filterLDS(features)
    # seperated, labels = seperate(features, 9, new_label)
    # labels = np.repeat(new_label, 80, axis=0)
    
    # print(features.shape, labels.shape)
    # save_folder = f'./data/NON_calibrated/{filename[:3]}'
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # generate_train_val_test_shuffle(features, labels, save_folder)
