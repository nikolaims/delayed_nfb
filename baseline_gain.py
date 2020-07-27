from settings import FB_ALL, BASELINE_BEFORE, BASELINE_AFTER
import pandas as pd
import seaborn as sns
from pingouin import rm_corr, mixed_anova, pairwise_ttests, rm_anova, plot_paired, friedman, anova, ttest
import pylab as plt
import numpy as np
from mne.stats import fdr_correction

sns.set_context("paper")
sns.set_style("dark")

threshold = 2.125


stats_file = 'baseline_block_stats_1channels_1bands_median_20ths.pkl'
stats_df_all = pd.read_pickle('data/{}'.format(stats_file))
# stats_df = stats_df.loc[stats_df.subj_id!=28]
stats_df_all = stats_df_all.loc[stats_df_all['block_number'].isin([BASELINE_AFTER, BASELINE_BEFORE])]
unique_blocks = list(stats_df_all['block_number'].unique())
stats_df_all = stats_df_all.loc[stats_df_all['threshold_factor'] == threshold]
stats_df_all['baseline'] = stats_df_all['block_number'].apply(lambda x: 'After' if x>10 else 'Before')

fb_types = ['FB0', 'FB250', 'FB500', 'FBMock']
stats_df_all = stats_df_all.loc[stats_df_all['fb_type'].isin(fb_types)]

metric_type='n_spindles'
res = mixed_anova(stats_df_all.query('metric_type=="{}"'.format(metric_type)), dv='metric', within='baseline', subject='subj_id', between='fb_type')

res2 = pairwise_ttests(stats_df_all.query('metric_type=="{}"'.format(metric_type)), dv='metric', within='baseline', subject='subj_id', between='fb_type', padjust='fdr_bh')


fig, axes = plt.subplots(2, 2, figsize=(9,4))
metric_types = ['magnitude', 'n_spindles', 'amplitude', 'duration']

p_all = np.zeros((4, 4))
for j_metric_type, metric_type in enumerate(metric_types):

    df_metric_type = stats_df_all.query('metric_type=="{}"'.format(metric_type))
    for j_fb_type, fb_type in enumerate(fb_types):
        ax = axes[j_metric_type//2, j_metric_type%2]
        df = df_metric_type.query('fb_type=="{}"'.format(fb_type))



        pd.set_option('display.max_columns', 500)
        res = ttest(df.query('baseline=="After"')['metric'], df.query('baseline=="Before"')['metric'], paired=True)
        # res = pairwise_ttests(df, dv='metric', within='baseline', subject='subj_id')
        p = res['p-val'].values[0]
        p_all[j_fb_type, j_metric_type] = p
        res_str = '$p_u$={:.3f}\n'.format(p) + r'$Diff_{CI95}$=' + '[{}, {}]'.format(*res['CI95%'].values[0])

        x_before = df.query('baseline=="Before"')['metric'].values
        x_after = df.query('baseline=="After"')['metric'].values
        for j in range(len(x_before)):
            pair = np.array([x_before[j], x_after[j]])
            ax.plot(np.array([0, 2]) + 3*j_fb_type, pair, '--o', color='C3' if p<0.05 else 'k', linewidth=0.75, markersize=2, alpha=0.7)
        ax.text(0+3*j_fb_type, 1.1*(df_metric_type['metric'].max()-df_metric_type['metric'].min()) + df_metric_type['metric'].min(), res_str, size=6, color='C3' if p < 0.05 else 'k')


        # ax.set_yticks([])
        ax.set_title(['A. ', 'B. ', 'C. ', 'D. '][j_metric_type]+metric_type+'\n\n')

for ax in axes.ravel():
    ax.set_xticks(np.arange(3*4))
    ax.set_xticklabels(np.concatenate([[r'${}^{Before}$', fb_t, '${}^{After}$'] for fb_t in fb_types]), size=7)
    [ax.axvline(k, color='w', zorder=-100) for k in range(1, 3*4, 3)]
            # ax.set_xlim([-0.5, 1.5])
plt.tight_layout()
fig.savefig('results/6_baselines_comps_threshold_factor_2_125.png', dpi=250)
fdr_correction(p_all[:, 3])


metric_type='magnitude'
stats_df_all['ratio'] = np.nan
for group, df in stats_df_all.groupby('subj_id'):
    dev = df.loc[df['baseline']=='After', 'metric'].values/df.loc[df['baseline']=='Before', 'metric'].values
    stats_df_all.loc[(stats_df_all['baseline']=='Before') & (stats_df_all['subj_id']==group), 'ratio'] = dev
stats_df_ratio = stats_df_all.dropna()
pairwise_ttests(stats_df_ratio.query('metric_type=="{}"'.format(metric_type)), dv='ratio', between='fb_type', padjust='fdr_bh')
anova(stats_df_ratio.query('metric_type=="{}"'.format(metric_type)), dv='ratio', between='fb_type')



# figure mut info
stats_df_all = pd.read_pickle('data/{}'.format(stats_file))
fb_types = ['FB0', 'FB250', 'FB500', 'FBMock']
stats_df_all = stats_df_all.loc[stats_df_all['fb_type'].isin(fb_types)]
# stats_df = stats_df.loc[stats_df.subj_id!=28]
stats_df_all = stats_df_all.loc[stats_df_all['block_number'].isin([BASELINE_AFTER, BASELINE_BEFORE])]
unique_thresholds= stats_df_all['threshold_factor'].unique()
fig = plt.figure(figsize=(4,3))
from sklearn.feature_selection import mutual_info_regression as mi
mi_list = []
for th in unique_thresholds:
    data = stats_df_all.query('threshold_factor=={}'.format(th))
    amp = data.query('metric_type=="amplitude"')['metric'].values
    dur = data.query('metric_type=="duration"')['metric'].values
    n_s = data.query('metric_type=="n_spindles"')['metric'].values
    np.hstack((mi(amp[:, None], n_s), mi(n_s[:, None], dur), mi(dur[:, None], amp)))

    mi_list.append(np.hstack((mi(amp[:, None], n_s), mi(n_s[:, None], dur), mi(dur[:, None], amp))))

plt.plot(unique_thresholds, mi_list)
plt.plot(unique_thresholds, np.mean(mi_list, 1), '--k')
plt.legend(['n_spindles - amplitude', 'n_spindles - duration', 'amplitude - duration', 'average MI'])
plt.xlabel('Threshold factor, $\mu$')
plt.ylabel('Mutual information')
plt.scatter(unique_thresholds[np.argmin(np.mean(mi_list, 1))], np.min(np.mean(mi_list, 1)), color='C3', zorder=100)
plt.plot([unique_thresholds[np.argmin(np.mean(mi_list, 1))]]*2, [0, np.min(np.mean(mi_list, 1))], '--', color='C3', zorder=100)
plt.ylim(0, plt.ylim()[1])
plt.tight_layout()
# fig.savefig('release/results/1_best_threshold_by_mutual_info.png', dpi=250)