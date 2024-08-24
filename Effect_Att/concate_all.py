import numpy as np
import os

all_data = []
all_label = []
for i in range(1, 11):
    if i == 10:
        data_path = f'./Data/len_96/False/s{i}/'
    else:
        data_path = f'./Data/len_96/False/s0{i}/'

    data = np.load(os.path.join(data_path, 'data.npy'))
    print(f'./Data/len_96/False/s{i}/', data.shape)
    label = np.load(os.path.join(data_path, 'label.npy'))
    index = range(0, 3200, 80)
    print(label[index])
    print(f'./Data/len_96/False/s{i}/', label.shape)
    all_data.append(data)
    all_label.append(label)

np_data = np.array(all_data)
np_data = np_data.reshape((-1, 32, 4, 96))
np_label = np.array(all_label).reshape((32000, ))

savepath = './Data/len_96/cross10/'
if not os.path.exists(savepath):
    os.makedirs(savepath)
np.save(os.path.join(savepath, "data.npy"), np_data)
np.save(os.path.join(savepath, "label.npy"), np_label)
