from settings import FB_ALL
import pandas as pd
import numpy as np
from stats.fan98test import eval_z_score, simulate_h0_distribution, get_p_val_one_tailed, adaptive_neyman_test, legendre_transform, identity_transform, corrcoef_test, get_p_val_two_tailed, legendre_projector
import pylab as plt
import seaborn as sns
from mne.stats import fdr_correction
from scipy.stats import t as tdist
from  scipy.stats import shapiro
from pingouin import rm_corr
sns.set_context("paper")
sns.set_style("dark")
# sns.set_style("dark", {"axes.facecolor": ".9"})


# STAT_FUN = corrcoef_test
# TRANSFORM_FUN = identity_transform
# P_VAL_FUN = get_p_val_two_tailed

STAT_FUN = adaptive_neyman_test
TRANSFORM_FUN = legendre_transform
P_VAL_FUN = get_p_val_one_tailed
PLOT_Z_SCORE_OPT_PROJ = True

stats_file = 'block_stats_1channels_1bands_median_20ths.pkl'
stats_df_all = pd.read_pickle('data/{}'.format(stats_file))
# stats_df = stats_df.loc[stats_df.subj_id!=28]
stats_df_all = stats_df_all.loc[stats_df_all['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df_all['block_number'].unique())
stats_df_all['k'] = stats_df_all['block_number'].apply(lambda x: unique_blocks.index(x))

metric_type = 'amplitude'
unique_thresholds = stats_df_all['threshold_factor'].unique()
fb_types = ['FB0', 'FB250', 'FB500', 'FBMock'][::-1]
stats_df_all = stats_df_all.loc[stats_df_all['fb_type'].isin(fb_types)]


threshold = 2.5
threshold_index = np.where(unique_thresholds == threshold)[0][0]


comps = []
for ind1 in range(len(fb_types)):
    for ind2 in range(len(fb_types) - 1, ind1, -1):
        comps.append(fb_types[ind2] + ' - ' + fb_types[ind1])

stats_all_metrics = {}
stats_extra_all_metrics = {}
p_vals_all_metrics = {}
z_scores_all_metrics = {}
metric_types = ['magnitude', 'n_spindles', 'duration', 'amplitude']

fig, axes = plt.subplots(1, 4, figsize=(6, 3), sharey=True, sharex=True)
fig2, axes2 = plt.subplots(3, 4, figsize=(6, 7), sharey='row', sharex=True)
all_ax = np.concatenate([axes, axes2.ravel()])
[ax.axvline(5, color='w', linewidth=0.75, zorder=-100) for ax in all_ax]
[ax.axvline(10, color='w', linewidth=0.75, zorder=-100) for ax in all_ax]
[ax.axhline(1, color='w', linewidth=0.75, zorder=-100) for ax in all_ax]
[ax.set_xlabel('block') for ax in np.concatenate([axes, axes2[-1, :]])]
[ax.set_title(title) for ax, title in zip(axes, fb_types[::-1])]
[ax.set_title(title) for ax, title in zip(axes2[0, :], fb_types[::-1])]
fig.subplots_adjust(bottom=0.300, wspace=0.075)
fig2.subplots_adjust(bottom=2*0.300/5, wspace=0.075)


metric_types_ax = {'n_spindles': 0, 'duration': 1, 'amplitude': 2}
fb_typs_axes = {'FB0': 0, 'FB250': 1, 'FB500': 2, 'FBMock': 3}
fb_typs_colors = {'FB0': 'C0', 'FB250': 'C2', 'FB500': 'C1', 'FBMock': 'C3'}
metric_types_lims = {'magnitude': np.array([0.65, 1.35]), 'n_spindles': np.array([0.1, 1.9]),
                     'duration': np.array([0.3, 1.7]), 'amplitude': np.array([0.7, 1.30])}

metrics_df = pd.DataFrame(columns=['subj_id', 'fb_type', 'k', 'metric_type', 'metric'])

shapiro_p_vals = []
shapiro_names = []
shapiro_samples = []
for metric_type in metric_types:
    stats_all_th = []
    p_all_th = []
    stats_extra_all_th = []

    z_scores_all_th = []
    for th in (unique_thresholds if metric_type != 'magnitude' else unique_thresholds[:1]):
        stats_df = stats_df_all.query('threshold_factor=={} & metric_type=="{}"'.format(th, metric_type))
        fb_data_points = []
        for ind, fb_type in enumerate(fb_types):
            fb_stats_df = stats_df.query('fb_type=="{}"'.format(fb_type))
            unique_subj = fb_stats_df.subj_id.unique()
            data_points = np.zeros((len(unique_subj), len(FB_ALL)))
            for j, subj_id in enumerate(unique_subj):
                data_points[j, :] = fb_stats_df.query('subj_id=={}'.format(subj_id))['metric']
                data_points[j, :] /= data_points[j, :].mean()
                if th == threshold or metric_type == 'magnitude':
                    metrics_df = metrics_df.append(pd.DataFrame(
                        {'subj_id': subj_id, 'fb_type': fb_type, 'k': np.arange(len(unique_blocks)),
                         'metric_type': metric_type, 'metric': data_points[j,:]}))
            # if th == threshold or metric_type == 'magnitude':
            shapiro_p_vals.append([shapiro(x)[1] for x in data_points.T])
            shapiro_samples.append([x for x in data_points.T])
            shapiro_names.append(['{} {} {} {}'.format(metric_type, fb_type, k, th) for k in range(1, 16)])

            fb_data_points.append(data_points[:, :])
            ax = (axes[fb_typs_axes[fb_type]]
                  if metric_type=='magnitude' else axes2[metric_types_ax[metric_type], fb_typs_axes[fb_type]])
            if th == threshold or metric_type=='magnitude':
                ax.errorbar(np.arange(len(unique_blocks))+1, data_points.mean(0),
                            2*data_points.std(0) / np.sqrt(data_points.shape[0]), color=fb_typs_colors[fb_type],
                            linewidth=0.75, elinewidth=0.75, linestyle='', marker='o', markersize=2, label=fb_type, zorder=100)
                ax.plot(np.arange(len(unique_blocks))+1, data_points.T, color=fb_typs_colors[fb_type],
                        linewidth=0.5, linestyle='-', markersize=2, label=fb_type, alpha=0.25)

                df = metrics_df.query('metric_type=="{}" & fb_type=="{}"'.format(metric_type, fb_type))
                df['k'] = df['k'].astype(int)
                s = r'$\rho_{CI95}$=' + rm_corr(df, 'k', 'metric', 'subj_id')['CI95%'].values[0]

                lim = metric_types_lims[metric_type]
                ax.text(1, lim[0] + (lim[1]-lim[0])*0.9, s, size=7)
                if fb_typs_axes[fb_type] == 0: ax.set_ylabel(metric_type)


                ax.set_ylim(*lim)
                ax.set_yticks([lim[0], 1., lim[1]])


        stats = []
        stats_extra = []
        z_scores = []
        ds = []
        ps = []
        for comp in comps:
            ind1, ind2 = [fb_types.index(fb_type) for fb_type in comp.split(' - ')]
            z_score, d = eval_z_score(fb_data_points[ind1], fb_data_points[ind2])
            stat, stat_extra = STAT_FUN(TRANSFORM_FUN(z_score), d, return_extra=True)
            h0_distribution = simulate_h0_distribution(n=len(unique_blocks), d=d, transform=TRANSFORM_FUN,
                                                       stat_fun=STAT_FUN, verbose=False, sim_verbose=True)
            ps.append(P_VAL_FUN(stat, h0_distribution))
            stats.append(stat)
            stats_extra.append(stat_extra)
            z_scores.append(z_score)
            ds.append(d)
        stats_all_th.append(stats)
        p_all_th.append(ps)
        stats_extra_all_th.append(stats_extra)
        z_scores_all_th.append(z_scores)


    stats_all_th = np.array(stats_all_th).T
    _, p_corrected = fdr_correction(np.array(p_all_th).T)
    stats_all_metrics[metric_type] = stats_all_th
    p_vals_all_metrics[metric_type] = p_corrected
    z_scores_all_metrics[metric_type] = np.array(z_scores_all_th)
    stats_extra_all_metrics[metric_type] = np.array(stats_extra_all_th)

# print not-normal samples
shapiro_p_vals = np.array(shapiro_p_vals).ravel()
fdr_p_shapiro = fdr_correction(shapiro_p_vals)
shapiro_names = np.array(shapiro_names).ravel()
print('FDR shapiro', shapiro_names[fdr_p_shapiro[0]])
print('Bonferroni shapiro', shapiro_names[shapiro_p_vals < 0.05/len(shapiro_p_vals)])
# plt.figure()
# [plt.scatter(x, k*np.ones_like(x), alpha=0.2, color='k') for k, x in enumerate(np.array(shapiro_samples).flatten()[fdr_p_shapiro[0]])]
# plt.yticks(np.arange(len(shapiro_names[fdr_p_shapiro[0]])), shapiro_names[fdr_p_shapiro[0]])
# plt.tight_layout()

fig.savefig('results/2a_magnitude_avg_vs_block.png', dpi=250)
fig2.savefig('results/2b_metric_avg_vs_block.png', dpi=250)


# figure: p-val vs threshold vs metric
fig = plt.figure(constrained_layout=True, figsize=(6, 2.5))
axes = []
gs = fig.add_gridspec(1, 7)
axes.append(fig.add_subplot(gs[0]))
for k in range(3):
    axes.append(fig.add_subplot(gs[k*2+1:k*2+3]))

# p_corrected = p
xticklabels0 = [th if th % 0.5 == 0 else '' for th in unique_thresholds]
# fig, axes = plt.subplots(1, len(metric_types))
for ax, metric_type in zip(axes, metric_types):
    p_corrected = p_vals_all_metrics[metric_type]

    if metric_type == 'magnitude':
        xticklabels = []
        yticklabels = comps
    else:
        xticklabels = xticklabels0
        yticklabels = ['']*len(comps)

    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6]) if metric_type == metric_types[-1] else None
    ax = sns.heatmap(np.log10(p_corrected), mask=p_corrected>0.05, yticklabels=yticklabels, xticklabels=xticklabels,
                     cmap='Blues_r', ax=ax, cbar=metric_type == metric_types[-1], vmax=0, vmin=-4, linewidths=0.5,
                     cbar_ax=cbar_ax, cbar_kws={'ticks': np.log10([0.05, 0.01, 0.001, 0.0001])})
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90 )
    ax.set_xlabel('th. factor' if metric_type != 'magnitude' else '')
    ax.set_ylim(len(comps), 0)
    ax.set_title(metric_type)

    if metric_type!='magnitude':
        ax.axvline(12, color='C3', linewidth=1, linestyle='-', alpha=0.5)
        ax.axvline(13, color='C3', linewidth=1, linestyle='-', alpha=0.5)

cbar_ax.set_yticklabels([0.05, 0.01, 0.001, 0.0001])
cbar_ax.set_title('p-value\n(FDR)')
cbar_ax.fill_between([-10, 1], np.log10([0.05] * 2), [0.] * 2, color='#EAEAF2')
plt.subplots_adjust(left = 0.2, right=0.8, bottom=0.2, top=0.8)
fig.savefig('results/4_significance_heatmap.png', dpi=250)


# figure z_scores

comps_colors = ['C0', 'C2', 'C1', 'C4', 'C4', 'C6']
fig, axes = plt.subplots(len(comps) + 1, len(metric_types), figsize=(6, 6))
fig2, axes2 = plt.subplots(3, 3, figsize=(5, 4))

for ax in np.concatenate([axes.ravel(), axes2.ravel()]):
    ax.set_ylim(-5, 5)
    ax.set_xticks([5, 10, 15])
    ax.set_xlim(-0, 16)
    ax.axvline(5, color='w', zorder=-100, linewidth=0.75)
    ax.axvline(10, color='w', zorder=-100, linewidth=0.75)
    ax.axhline(0, color='w', zorder=-100, linewidth=0.75)
    ax.set_yticks([])

for ax in np.concatenate([axes[-1, :], axes2[:, -1]]):
    ax.set_xlabel('block')
    ax.set_ylim(-3, 3)

axes[-1, 0].set_yticks([1])
axes[-1, 0].set_yticklabels(['Difference\nsplines'])


for j_metric_type, metric_type in enumerate(metric_types):
    p_corrected = p_vals_all_metrics[metric_type]
    z_score = z_scores_all_metrics[metric_type]
    stats_extra = stats_extra_all_metrics[metric_type]
    th_index = threshold_index if metric_type != 'magnitude' else 0
    axes[0, j_metric_type].set_title(metric_type)


    for j_comp, comp in enumerate(comps):
        ax_list = [axes[j_comp, j_metric_type]] + ([axes2[j_comp%3, j_comp//3]] if metric_type =='magnitude' else [])
        for ax in ax_list:
            significant = p_corrected[j_comp, th_index] < 0.05
            # if significant:
            #     ax.set_facecolor('#eee9e9')

            best_z_score = z_score[th_index, j_comp]
            ax.plot(np.arange(len(unique_blocks))+1, best_z_score, 'o', markersize=1, linewidth=0.5,
                    color=comps_colors[j_comp] if significant else '.2', zorder=1000)

            if PLOT_Z_SCORE_OPT_PROJ:
                best_m = stats_extra[th_index, j_comp]
                q = legendre_projector(len(unique_blocks))
                proj = best_z_score.dot(q)
                proj[best_m:] = 0
                proj = proj.dot(q.T)
                ax.plot(np.arange(len(unique_blocks)) + 1, proj,  color=comps_colors[j_comp] if significant else 'k',
                        markersize=2, linewidth=0.7 if significant else 0.5, linestyle='-' if significant else '--', alpha=1 if significant else 0.4)
                ax.text(1, 3, 'p={:.4f} m={}'.format(p_corrected[j_comp, th_index], best_m), size=5,
                        color='C3' if significant else 'k')
                if significant:
                    ax2_list = [axes[-1, j_metric_type]] + ([axes2[1, 2]] if metric_type=='magnitude' else [])
                    for ax2 in ax2_list:
                        ax2.plot(np.arange(len(unique_blocks)) + 1, proj+0.3*(j_comp==1),
                                                     color=comps_colors[j_comp], label=comps_colors[j_comp], linewidth=0.7)


            else:
                ax.plot(np.nan)

            ax.axhline(tdist.ppf(0.975, ds[j_comp]), color='w', zorder=-100, linestyle='--', linewidth=0.75)
            ax.axhline(tdist.ppf(0.025, ds[j_comp]), color='w', zorder=-100, linestyle='--', linewidth=0.75)

            ax.set_xticks([])






            if j_metric_type == 0:
                ax.set_yticks([0])
                ax.set_yticklabels([comp])
            # if j_metric_type == 0 :
            #     ax.set_ylabel(comp, rotation=0, labelpad=30, size=8)
fig.subplots_adjust(left = 0.2, right=0.9, bottom=0.1, top=0.9, hspace=0.01, wspace=0.07)

for comp, ax in zip(comps, axes2[:, :2].T.ravel()):
    ax.set_yticks([])
    ax.set_title(comp)
axes2[0, 2].axis('off')
axes2[2, 2].axis('off')
axes2[1, 2].set_title('Difference splines')
for ax in [axes2[-1, 0], axes2[-1, 1]]:
    ax.set_xticks([5, 10, 15])
    ax.set_xlabel('block')

fig2.subplots_adjust(left = 0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.3, wspace=0.1)

fig.savefig('results/3_t_stats_vs_block.png', dpi=250)
fig2.savefig('results/3b_t_stats_vs_block_magnitude.png', dpi=250)
# fig2.savefig('release/results/3a_t_stats_vs_block_significant_splines.png', dpi=250)

# figure mut info
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
fig.savefig('results/1_best_threshold_by_mutual_info.png', dpi=250)

# fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
# for j, fb_type in enumerate(fb_types[::-1]):
#     ax = axes[j]
#     stats_df = stats_df_all.query('fb_type=="{}"'.format(fb_type))
#     mi_list = []
#     for th in unique_thresholds:
#         data = stats_df.query('threshold_factor=={}'.format(th))
#         amp = data.query('metric_type=="amplitude"')['metric'].values
#         dur = data.query('metric_type=="duration"')['metric'].values
#         n_s = data.query('metric_type=="n_spindles"')['metric'].values
#         np.hstack((mi(amp[:, None], n_s), mi(n_s[:, None], dur), mi(dur[:, None], amp)))
#
#         mi_list.append(np.hstack((mi(amp[:, None], n_s), mi(n_s[:, None], dur), mi(dur[:, None], amp))))
#
#     ax.plot(unique_thresholds, mi_list)
#     ax.plot(unique_thresholds, np.mean(mi_list, 1), '--k')
#     ax.set_title(fb_type)
#
#     ax.set_xlabel('threshold factor')
#     ax.set_ylabel('Mutual information')
#     ax.scatter(unique_thresholds[np.argmin(np.mean(mi_list, 1))], np.min(np.mean(mi_list, 1)), color='r', zorder=100)
#     if j == 3: ax.legend(['n_spin. vs ampl.', 'dur. vs n_spin.', 'ampl vs dur.', 'average mut. inf.'])
