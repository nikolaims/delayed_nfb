import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import h5py
import os
import mne
import pylab as plt
import seaborn as sns
sns.set_style()

# import nfb lab data loader
from utils import load_data
from utils import annotate_bad
from settings import CHANNELS_SEL, MNE_INFO

FLANKER_WIDTH = 2
GFP_THRESHOLD = 100e-6
bad_channels = {17: ['Pz'], 20: ['P7', 'P8', 'CP5'], 46: ['CP1'], 8: ['F3'], 35: ['P8'], 40: ['O1']}
bad_subjects = [18, 23, 33, 37, 38]

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0]
                 in ['FB0', 'FBMock', 'FB250', 'FB500', 'FBLow'])][:]

# store data
columns = ['subj_id', 'block_number'] + CHANNELS_SEL + ['PHOTO'] + ['online_envelope'] + ['is_not_bad']
probes_df = pd.DataFrame(columns=columns, dtype='float32')
datasets_df = pd.DataFrame(columns=['dataset', 'subj_id', 'band', 'fb_type', 'snr'])

# iter subjects
fig, axes = plt.subplots(10, 4, sharex=True)
for ax in axes.flatten():
    ax.set_yticks([])
    sns.despine(None, ax, True, True, True)
for ax in axes[-1, :]:
    ax.set_xlabel('Freq., Hz')
fb_types_count = {'FB0': 0, 'FB250': 0, 'FB500': 0,'FBMock': 0}
for subj_id, dataset in enumerate(datasets):
    # if subj_id > 1: continue
    if subj_id in bad_subjects: continue
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

    # load fb signal params
    with h5py.File(dataset_path) as f:
        eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
        band = f['protocol10/signals_stats/Alpha0/bandpass'].value

    # load data
    df, fs, channels, p_names = load_data(dataset_path)
    df['online_envelope'] = df['signal_Alpha0']

    # get FB type
    # fb_type = df.query('block_number==6')['block_name'].values[0]
    fb_type = info.loc[info['dataset']==dataset, 'type'].values[0]
    if fb_type == 'FBLow': continue

    # rename FB blocks to "FB"
    df['block_name'] = df['block_name'].apply(lambda x: 'FB' if ('FB' in x and 'Pause' not in x) else x)

    # drop pauses except PauseFB
    df = df.loc[df['block_name'].isin(['Baseline0', 'Close', 'Baseline', 'FB', 'PauseFB'])]

    # remove eyes artifacts ICA
    df[channels] = df[channels].values.dot(eye_rejection_matrix)

    # drop columns
    df = df.drop(columns=[ch for ch in channels if ch not in CHANNELS_SEL + ['PHOTO']])
    channels = CHANNELS_SEL + ['PHOTO']

    # GFP threshold arthifact segments
    th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
    df = df.loc[th < GFP_THRESHOLD]

    # down sample to 250
    ba = sg.butter(4, np.array([1, 100])/fs*2, 'band')
    df[channels[:-1]] = sg.filtfilt(*ba, df[channels[:-1]].values, axis=0) # except PHOTO
    # handle PHOTO
    y = sg.filtfilt(np.ones(20) / 20, [1], df['PHOTO'].values)
    y_down = pd.Series(y).rolling(1000, center=True).min().rolling(10000, center=True).mean().fillna(
        method='bfill').fillna(method='ffill').values
    df['PHOTO'] = y - y_down
    # handle envelope
    df['online_envelope'] = sg.filtfilt([0.5, 0.5], [1., 0.], df['online_envelope'].values, axis=0)
    df = df.iloc[::2]
    FS = 250

    # interpolate bad channels
    if subj_id in bad_channels:
        raw = mne.io.RawArray(df[CHANNELS_SEL].values.T, MNE_INFO)
        raw.info['bads'].extend(bad_channels[subj_id])
        df[CHANNELS_SEL] = raw.copy().interpolate_bads().get_data().T

    # annotate and delete bad segments
    mask_file_path = 'data/bad_annotations/1good_mask_s{}.npy'.format(subj_id)
    if os.path.exists(mask_file_path):
        mask = np.load(mask_file_path)
    else:
        mask, annotations = annotate_bad(df[CHANNELS_SEL], FS, CHANNELS_SEL, GFP_THRESHOLD)
        annotations.save('data/bad_annotations/s{}.csv'.format(subj_id))
        np.save(mask_file_path, mask)

    df['is_not_bad'] = mask.astype(int)
    # df = df.loc[mask]

    # estimate snr
    freq, pxx = sg.welch(df.loc[mask].query('block_name=="Close" | block_name=="Baseline0"')['P4'], FS, nperseg=FS * 4)
    sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
            (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    snr = sig / noise

    ax = axes[fb_types_count[fb_type], ['FB0', 'FB250', 'FB500', 'FBMock'].index(fb_type)]
    if fb_types_count[fb_type] == 0: ax.set_title(fb_type)
    fb_types_count[fb_type] += 1

    ax.plot(freq, pxx, 'k', zorder=200)

    # freq, pxx = sg.welch(df.loc[mask].query('block_name=="Baseline0"')['P4'], FS, nperseg=FS * 4)
    # sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    # noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
    #         (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    # snr = sig / noise
    #
    # ax.plot(freq, pxx, 'k', linewidth=0.7, zorder=200)

    ax.axvspan(*band, color='C3')
    ax.axvspan(band[0]-2, band[0], color='C0')
    ax.axvspan(band[1], band[1]+2, color='C0')
    ax.text(15, max(pxx)*0.5, '{:.1f}'.format(snr))
    ax.set_ylabel(str(subj_id))

    ax.set_xlim(0, 20)

    print(subj_id, fb_type)