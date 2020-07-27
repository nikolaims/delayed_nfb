import pandas as pd
import numpy as np
from tqdm import tqdm
from settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS, MEANINGFUL_BLOCKS, BASELINE_BEFORE, BASELINE_AFTER
from utils import band_hilbert

P4_ONLY = True
BASELINES_ONLY = False
USE_PERCENTILES = False
channels = (['P4'] if P4_ONLY else CHANNELS)
if USE_PERCENTILES:
    threshold_factors = np.arange(50, 100, 2.5)
else:
    threshold_factors = np.arange(1, 3.5, 0.125)
#
bands = dict(zip(['alpha'], [1]))
res_df_name = '{}block_stats_{}channels_{}bands_{}_{}ths'.format('baseline_' if BASELINES_ONLY else '', len(channels),
                                                                 len(bands), 'perc' if USE_PERCENTILES else 'median',
                                                                 len(threshold_factors))
print(res_df_name)

# load pre filtered data
probes_df = pd.read_pickle('data/preprocessed_eeg_p4_5_groups_clear_fs250Hz_prefilt1_100Hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')

# prepare data frame to save metrics
columns=['subj_id', 'channel', 'fb_type', 'metric', 'metric_type', 'block_number','threshold_factor', 'band']

for subj_id in datasets_df['subj_id'].values[:]:
    stats_df = pd.DataFrame(columns=columns)

    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]

    # subj band
    for band_name, band_factor in bands.items():
        band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0]) * band_factor

        # block numbers utils
        block_numbers = data['block_number'].values
        unique_block_numbers = np.unique(block_numbers)

        # select meaningful blocks (no pauses)
        unique_block_numbers = [u for u in unique_block_numbers if u in MEANINGFUL_BLOCKS]

        is_not_bad = (data['is_not_bad'] == 1).values
        block_numbers = block_numbers[is_not_bad]

        for ch in tqdm(channels, str(subj_id) + band_name):
            # channel data if channels is ICA get projection
            ch_data = data[ch].values

            # compute envelope
            env = np.abs(band_hilbert(ch_data, FS, band))
            env = env[is_not_bad]

            if BASELINES_ONLY:
                median = np.median(env[np.isin(block_numbers, [BASELINE_BEFORE, BASELINE_AFTER])])
                unique_block_numbers = [BASELINE_BEFORE, BASELINE_AFTER]
            else:
                median = np.median(env[np.isin(block_numbers, FB_ALL)])

            for block_number in unique_block_numbers:
                signal = env[block_numbers == block_number]

                # mean magnitude in uV
                magnitude_j = np.mean(signal) * 1e6

                # iterate thresholds factors
                for threshold_factor in threshold_factors:
                    if USE_PERCENTILES:
                        threshold = np.percentile(env[np.isin(block_numbers, FB_ALL)], threshold_factor)
                    else:
                        threshold = threshold_factor * median
                    #

                    # get spindles mask
                    spindles_mask = signal > threshold
                    if np.sum(np.diff(spindles_mask.astype(int)) == 1) > 0:

                        # number of spindles
                        n_spindles_j = np.sum(np.diff(spindles_mask.astype(int)) == 1)

                        # mean spindle duration
                        duration_j = np.sum(spindles_mask) / n_spindles_j / FS

                        # mean spindle amplitue
                        amplitude_j = np.mean(signal[spindles_mask]) * 1e6
                    else:
                        n_spindles_j = 1 # TODO replace by NaN
                        duration_j = 0.005
                        amplitude_j = threshold * 1e6

                    # save metrics above for channel
                    stats_df = stats_df.append(pd.DataFrame(
                        {'subj_id': subj_id, 'channel': ch, 'fb_type': fb_type,
                         'metric': [magnitude_j, n_spindles_j/(len(signal)/FS/60), duration_j, amplitude_j],
                         'metric_type': ['magnitude', 'n_spindles', 'duration', 'amplitude'],
                         'block_number': block_number, 'threshold_factor': threshold_factor, 'band': band_name}),
                        ignore_index=True)

    stats_df.to_pickle('data/temp/{}_{}.pkl'.format(res_df_name, subj_id))

stats_df = pd.DataFrame(columns=columns)
for subj_id in datasets_df['subj_id'].values[:]:
    stats_df = stats_df.append(pd.read_pickle('data/temp/{}_{}.pkl'.format(res_df_name, subj_id)))
stats_df.to_pickle('data/{}.pkl'.format(res_df_name))