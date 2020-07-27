from utils import Montage
import pickle
from mne import create_info


FS_RAW = 500
FS = 250
CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'C3', 'CZ', 'C4',
            'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'P4', 'P8', 'O1', 'OZ', 'O2', 'T7', 'PZ']

CHANNELS_SEL = ['F3', 'FZ', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7',
                'P3', 'P4', 'P8', 'O1', 'OZ', 'O2', 'PZ']

rename_chs_to_standard_1020 = lambda x: [ch.replace('Z', 'z').replace('FP', 'Fp') for ch in x]

BLOCK_NAMES = [None, 'Close', 'Baseline0', 'PauseBL', 'Baseline', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB',
               'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB',
               'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB',
               'PauseBL', 'Baseline']
FB_ALL = [k for k, name in enumerate(BLOCK_NAMES) if name=='FB']
MEANINGFUL_BLOCKS = [k for k, name in enumerate(BLOCK_NAMES) if name in ['FB', 'Close', 'Baseline', 'Baseline0']]
CLOSE = 1
OPEN = 2
BASELINE_BEFORE = 4
BASELINE_AFTER = len(BLOCK_NAMES) - 1
ICA_BLOCKS = [CLOSE, OPEN, BASELINE_BEFORE, FB_ALL[0]]
MONTAGE = Montage(CHANNELS)
ICA_CHANNELS = ['ICA{}'.format(k + 1) for k in range(len(CHANNELS))]


MNE_INFO = create_info(rename_chs_to_standard_1020(CHANNELS_SEL), FS, 'eeg', 'standard_1020')
# from mne.viz import plot_sensors
# plot_sensors(MNE_INFO, show_names=True)

def save_ica(ica, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(ica, output, pickle.HIGHEST_PROTOCOL)

def load_ica(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)