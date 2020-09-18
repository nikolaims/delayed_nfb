import mne
import numpy as np
import pylab as plt
import scipy.signal as sg
import xml.etree.ElementTree as ET
import h5py
import pandas as pd


def azimuthal_equidistant_projection(hsp):
    radius = 20
    width = 5
    height = 4
    # Calculate angles
    r = np.sqrt(np.sum(hsp ** 2, axis=-1))
    theta = np.arccos(hsp[:, 2] / r)
    phi = np.arctan2(hsp[:, 1], hsp[:, 0])

    # Mark the points that might have caused bad angle estimates
    iffy = np.nonzero(np.sum(hsp[:, :2] ** 2, axis=-1) ** (1. / 2)
                      < np.finfo(np.float).eps * 10)
    theta[iffy] = 0
    phi[iffy] = 0

    # Do the azimuthal equidistant projection
    x = radius * (2.0 * theta / np.pi) * np.cos(phi)
    y = radius * (2.0 * theta / np.pi) * np.sin(phi)

    pos = np.c_[x, y]
    return pos




class Montage(pd.DataFrame):
    CHANNEL_TYPES = ['EEG', 'MAG', 'GRAD', 'OTHER']

    def __init__(self, names, **kwargs):
        if not isinstance(names, list):
            super(Montage, self).__init__(names, **kwargs)
        else:
            super(Montage, self).__init__(columns=['name', 'type', 'pos_x', 'pos_y'])
            layout_eeg = Montage.load_layout('EEG1005')
            layout_mag = Montage.load_layout('Vectorview-mag')
            layout_mag.names = list(map(lambda x: x.replace(' ', ''), layout_mag.names))
            layout_grad = Montage.load_layout('Vectorview-grad')
            layout_grad.names = list(map(lambda x: x.replace(' ', ''), layout_grad.names))
            for name in names:
                if name.upper() in layout_eeg.names:
                    ch_ind = layout_eeg.names.index(name.upper())
                    self._add_channel(name, 'EEG', layout_eeg.pos[ch_ind][:2])
                elif name.upper() in layout_mag.names:
                    ch_ind = layout_mag.names.index(name.upper())
                    self._add_channel(name, 'MAG', layout_mag.pos[ch_ind][:2])
                elif name.upper() in layout_grad.names:
                    ch_ind = layout_grad.names.index(name.upper())
                    self._add_channel(name, 'GRAD', layout_grad.pos[ch_ind][:2])
                else:
                    self._add_channel(name, 'OTHER', (None, None))

    def _add_channel(self, name, type, pos):
        self.loc[len(self)] = {'name': name, 'type': type, 'pos_x': pos[0], 'pos_y': pos[1]}

    def get_names(self, type='ALL'):
        return list(self[self.get_mask(type)]['name'])

    def get_pos(self, type='ALL'):
        return (self[self.get_mask(type)][['pos_x', 'pos_y']]).values

    def get_mask(self, type='ALL'):
        if type in self.CHANNEL_TYPES:
            return (self['type'] == type).values
        elif type == 'ALL':
            return (self['type'] == self['type']).values
        elif type == 'GRAD2':
            return ((self['type'] == 'GRAD') & (self['name'].apply(lambda x: x[-1]) == '2')).values
        elif type == 'GRAD3':
            return ((self['type'] == 'GRAD') & (self['name'].apply(lambda x: x[-1]) == '3')).values
        else:
            raise TypeError('Bad channels type')

    @staticmethod
    def load_layout(name):
        if name == 'EEG1005':
            if int(mne.__version__.split('.')[1]) >= 19:  # validate mne version (mne 0.19+)
                layout = mne.channels.make_standard_montage('standard_1005')
                layout.names = layout.ch_names
                ch_pos_dict = layout._get_ch_pos()
                layout.pos = np.array([ch_pos_dict[name] for name in layout.names])
                layout.pos = azimuthal_equidistant_projection(layout.pos)
            else:
                layout = mne.channels.read_montage('standard_1005')
                layout.pos = azimuthal_equidistant_projection(layout.pos)
                layout.names = layout.ch_names
        else:
            layout = mne.channels.read_layout(name)
        layout.names = list(map(str.upper, layout.names))
        return layout

    def make_laplacian_proj(self, type='ALL', n_channels=4):
        pos = self.get_pos(type)
        proj = np.eye(pos.shape[0])
        for k in range(pos.shape[0]):
            proj[k, np.argsort(((pos[k] - pos) ** 2).sum(1))[1:1+n_channels]] = -1 / n_channels
        return proj

    def combine_grad_data(self, data):
        names = self.get_names('GRAD')
        grad2_mask = list(map(lambda x: x[-1]=='2', names))
        grad3_mask = list(map(lambda x: x[-1] == '3', names))
        combined_data = (data[grad2_mask]**2 + data[grad3_mask]**2)**0.5
        return combined_data, self.get_pos('GRAD')[grad2_mask]


def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x

class CFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, **kwargs):
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * fs < band[0]) | (w / n_fft * fs > band[1])] = 0
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        if weights is None:
            self.b = F.T.conj().dot(H)/n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b)-1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y


def annotate_bad(data, fs, channels, threshold):
    th = (np.abs(data).rolling(int(fs)//2, center=True).max().max(1).fillna(0).values > threshold).astype(int)
    onsets = np.where(np.diff(th)>0)[0]/fs
    durations = np.where(np.diff(th)<0)[0]/fs - onsets
    gfp_ann = mne.Annotations(onsets, durations, 'BAD')

    mne_info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data.values.T, mne_info)
    raw.set_annotations(gfp_ann)
    fig = raw.plot(n_channels=32, duration=30, scalings={'eeg': 60e-6})
    fig.canvas.key_press_event('a')
    plt.show(block=True)
    plt.close()

    good_mask = np.ones(len(data))
    t = np.arange(len(data))/fs
    for onset, duration in zip(raw.annotations.onset, raw.annotations.duration):
        good_mask[(t >= onset) & (t <= onset+duration)] = 0
    return good_mask > 0, raw.annotations



def _get_channels_and_fs(xml_str_or_file):
    root = ET.fromstring(xml_str_or_file)
    if root.find('desc').find('channels') is not None:
        channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    else:
        channels = [k.find('name').text for k in root.find('desc').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs


def _get_signals_list(xml_str):
    root = ET.fromstring(xml_str)
    derived = [s.find('sSignalName').text for s in root.find('vSignals').findall('DerivedSignal')]
    composite = []
    if root.find('vSignals').findall('CompositeSignal')[0].find('sSignalName') is not None:
        composite = [s.find('sSignalName').text for s in root.find('vSignals').findall('CompositeSignal')]
    return derived + composite


def _get_info(f):
    if 'channels' in f:
        channels = [ch.decode("utf-8")  for ch in f['channels'][:]]
        fs = f['fs'].value
    else:
        channels, fs = _get_channels_and_fs(f['stream_info.xml'][0])
    signals = _get_signals_list(f['settings.xml'][0])
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    block_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    return fs, channels, block_names, signals


def load_data(file_path):
    with h5py.File(file_path) as f:
        # load meta info
        fs, channels, p_names, signals = _get_info(f)

        # load raw data
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
        df = pd.DataFrame(np.concatenate(data), columns=channels)

        # load signals data
        signals_data = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
        df_signals = pd.DataFrame(np.concatenate(signals_data), columns=['signal_'+s for s in signals])
        df = pd.concat([df, df_signals], axis=1)

        # load timestamps
        if 'timestamp' in df:
            timestamp_data = [f['protocol{}/timestamp_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['timestamps'] = np.concatenate(timestamp_data)

        # events data
        events_data = [f['protocol{}/mark_data'.format(k + 1)][:] for k in range(len(p_names))]
        df['events'] = np.concatenate(events_data)

        # set block names and numbers
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
    return df, fs, channels, p_names