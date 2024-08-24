import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from scipy.signal import welch, butter, lfilter
from pykalman import KalmanFilter


# Deap description里没有提到降噪，只说了去artifacts，但其实带通滤波和ICA去artifacts已经包含一部分降噪功能了

# coeffs = pywt.wavedec(eeg_signal, 'db4', level=4) 可以用小波分解再平滑一些，或者EM平滑/LDS平滑


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


def feature_extraction2(filtered_data, N=32):
    filtered = np.zeros(filtered_data.shape)
    for m in range(N):
        kf = KalmanFilter(initial_state_mean=np.mean(filtered_data[:, m]), observation_covariance=1,)
        (filtered_c, _) = kf.smooth(filtered_data[:, m])
        filtered[:, m] = filtered_c[:, 0]

    return filtered


def LDS_filter(all_data):
    # all data shape: trial, channel, time, band
    trial, channel, time, band = all_data.shape
    filtered = np.zeros(all_data.shape)
    for t in range(trial):
        for chan in range(channel):
            for b in range(band):
                print("t, chan, b:", t, chan, b)
                kf = KalmanFilter(initial_state_mean=np.mean(all_data[t, chan, :, b]), observation_covariance=10,
                                  transition_covariance=0.1)
                (filtered_c, _) = kf.smooth(all_data[t, chan, :, b])
                filtered[t, chan, :, b] = filtered_c[:, 0]
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


if __name__ == '__main__':
    dataset_path = '../DSTS_attention/data/data_preprocessed_matlab/'
    window_len = 96
    smooth = False
    n = int(7680 / window_len)
    if n != (7680 / window_len):
        raise ValueError
    for filename in os.listdir(dataset_path):
        save_folder = f'./Data/len_{window_len}/{smooth}/{filename[:3]}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_path = os.path.join(dataset_path, filename)
        raw_data, new_label = load_data_label(file_path, how='all')
        labels = np.repeat(new_label, n, axis=0)
        filtered = filter_data(raw_data)  # band pass filter
        if smooth:
            filtered = LDS_filter(filtered)
            np.save(os.path.join(save_folder, "LDS_filtered.npy"), filtered)
        temp_filtered = filtered.transpose((0, 1, 3, 2))  # (40, 32, 4, 7680)
        temp_filtered = temp_filtered.reshape((40, 32, 4, window_len, n))  # (40, 32, 4, 12, 640)
        temp_filtered = temp_filtered.transpose((0, 4, 1, 2, 3))  # (40, 640, 32, 4, 12)
        filtered = temp_filtered.reshape((-1, 32, 4, window_len))  # (25600, 32, 4, 12)
        print(f'{filename} : data shape-{filtered.shape}, label shape-{labels.shape}')
        # print(labels)
        # x_train, x_test, y_train, y_test = train_test_split(filtered, labels, test_size=0.2, random_state=42,
        #                                                     shuffle=False)
        # print(y_test)
        np.save(os.path.join(save_folder, "data.npy"), filtered)
        np.save(os.path.join(save_folder, "label.npy"), labels)
        # splitTrainTest(filtered, labels, save_folder)
        # np.save(os.path.join(save_folder, "X_train.npy"), x_train)
        # np.save(os.path.join(save_folder, "X_test.npy"), x_test)
        # np.save(os.path.join(save_folder, "y_train.npy"), y_train)
        # np.save(os.path.join(save_folder, "y_test.npy"), y_test)  # x shape:  (25600, 32, 4, 12) , y shape:  (25600,)
