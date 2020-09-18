import pandas as pd
import numpy as np
from tqdm import tqdm
from settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS, MEANINGFUL_BLOCKS, BASELINE_BEFORE, BASELINE_AFTER
from utils import band_hilbert
import pylab as plt


# load pre filtered data
probes_df = pd.read_pickle('data/preprocessed_eeg_p4_5_groups_clear_fs250Hz_prefilt1_100Hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')

subj_id = 1
delays_df = pd.DataFrame(columns=['subj_id', 'fb_type', 'corr', 'delay', 'snr'])
for subj_id in datasets_df['subj_id'].values[:]:

    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]
    snr = datasets_df.query('subj_id=={}'.format(subj_id))['snr'].values[0]

    # subj band
    band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0])

    # block numbers utils
    block_numbers = data['block_number'].values
    unique_block_numbers = np.unique(block_numbers)

    # select meaningful blocks (no pauses)
    unique_block_numbers = [u for u in unique_block_numbers if u in FB_ALL]

    is_not_bad = (data['is_not_bad'] == 1).values
    block_numbers = block_numbers[is_not_bad]

    ch = "P4"

    ch_data = data[ch].values
    photo_data = data["PHOTO"].values

    # compute envelope
    env = np.abs(band_hilbert(ch_data, FS, band))
    env = env[is_not_bad]
    photo_data = photo_data[is_not_bad]

    signal = env[np.isin(block_numbers, unique_block_numbers)]
    photo = photo_data[np.isin(block_numbers, unique_block_numbers)]
    corrs = [np.corrcoef(signal[:-d], photo[d:])[0, 1] for d in range(1, FS)]
    delay_ind = np.argmax(corrs)
    corr = corrs[delay_ind]
    delay = delay_ind/FS*1000
    print(corr, delay)
    plt.scatter(delay, corr, c={'FB0': 'C0', 'FB250': 'C2', 'FB500': 'C1', 'FBMock': 'C3'}[fb_type])
    delays_df = delays_df.append({'subj_id': subj_id, 'delay': delay, 'corr': corr, 'fb_type': fb_type, 'snr': snr}, ignore_index=True)

plt.xlabel('Delay, ms')
plt.ylabel('Corr.')

delays_df.to_csv('data/delays.csv')

import seaborn as sns
sns.pairplot(delays_df[['fb_type', 'snr', 'corr', 'delay']], 'fb_type')