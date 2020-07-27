import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
sns.set_context('paper')
sns.set_style('dark')

fb_types = ['FB0', 'FB250', 'FB500', 'FBMock']
df = pd.read_pickle('data/info_allsubjs.pkl')
df = df.loc[df.fb_type.isin(fb_types)]

plt.figure(figsize=(2.5, 2))
ax = sns.swarmplot('fb_type', 'snr', data=df, order=fb_types, hue='fb_type', hue_order=['FB0', 'FB500', 'FB250', 'FBMock'])
ax.get_legend().set_visible(False)
plt.tight_layout()
plt.savefig('results/7_snr.png', dpi=200)