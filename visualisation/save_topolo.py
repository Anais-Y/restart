import mne
import torch
import numpy as np
import matplotlib.pyplot as plt


i=1
gamma_feats = torch.load(f'/data/Anaiis/garage/vis_data/7_20131030/gamma_feats_{i}.pt')
gamma_feats = gamma_feats.cpu().numpy()
gamma_feats = gamma_feats.reshape((-1, 62, 512))

n_channels = 62
sfreq = 256
n_times = 512
data = gamma_feats[0, :, :].squeeze()
# 设置通道名称（假设使用国际10-20系统的标准名称）
biosemi_montage = mne.channels.make_standard_montage('standard_1020')
# print(biosemi_montage.ch_names, biosemi_montage)
channel_names = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
    'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 
    'PO8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
]
print(len(channel_names))
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
evoked = mne.EvokedArray(data, info)
evoked.set_montage(biosemi_montage)
fig, ax = plt.subplots(1, 1, figsize=(6, 6)) 
mne.viz.plot_topomap(np.mean(evoked.data, axis=1), evoked.info, show=False, 
                    ch_type='eeg', cmap='Spectral_r', contours=15, axes=ax)#vlim=(vmin, vmax),
plt.show()