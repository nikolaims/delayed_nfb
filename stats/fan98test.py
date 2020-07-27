import numpy as np
import os


def eval_z_score(data_points1, data_points2):
    # mean, std and number of samples
    x1_mean = np.mean(data_points1, 0)
    x1_std = np.std(data_points1, 0)
    x1_n = data_points1.shape[0]
    x2_mean = np.mean(data_points2, 0)
    x2_std = np.std(data_points2, 0)
    x2_n = data_points2.shape[0]

    # degree of freedom(see page 1011, left top paragraph)
    d = x1_n + x2_n - 2

    # z - score over time(see formula 13)
    z_score = (x1_mean - x2_mean) / (x1_std ** 2 / x1_n + x2_std ** 2 / x2_n) ** 0.5
    return z_score, d


def eval_f_stat(data_points):
    n_all_sample = sum([group_data[:, 0].size for group_data in data_points])
    n_groups = len(data_points)
    n_blocks = data_points[0].shape[1]
    n_subjects_list  = [data_points[k].shape[0] for k in range(n_groups)]
    n_subjects = sum(n_subjects_list)
    f_stats_list = []
    for b in range(n_blocks):
        data = [group_data[:, b] for group_data in data_points]

        overall_mean = sum([group_data.sum() for group_data in data])/n_subjects
        group_means_list = [group_data.mean() for group_data in data]
        between_group_var = sum([n * (group_mean - overall_mean) ** 2
                                 for n, group_mean in zip(n_subjects_list, group_means_list)]) / (n_groups - 1)
        within_group_var = sum([np.sum((group_data - group_mean)**2)
                            for group_data, group_mean in zip(data, group_means_list)]) / (n_all_sample - n_groups)
        f_stat = between_group_var /(within_group_var + 1e-10)
        f_stats_list.append(f_stat)
    d1 = n_groups - 1
    d2 = n_all_sample - n_groups
    return f_stats_list, d1, d2


def adaptive_neyman_test(z_star, d, return_extra=False):

    # eval statistic for each number of first blocks (see fomula 6)
    T_an = np.zeros(len(z_star))
    d_factor = 1 if d is None else ((d - 2) ** 2 * (d - 4) / (d ** 2 * (d - 1))) ** 0.5
    for m in range(len(z_star)):
        T_an[m] = np.sum(z_star[:m + 1] ** 2 - 1) * d_factor / (2 * (m + 1)) ** 0.5

    # find maximum T (see formula 6)
    stat = np.max(T_an)

    # compute final stat(see fomula 4)
    loglogn = np.log(np.log(len(z_star)))

    stat = (2 * loglogn) ** 0.5 * stat - (2 * loglogn + 0.5 * np.log(loglogn) - 0.5 * np.log(4 * np.pi))

    if return_extra:
        opt_m = np.argmax(T_an) + 1
        return stat, opt_m
    return stat


def corrcoef_test(z_star, d, return_extra=False):
    stat = np.corrcoef(np.arange(len(z_star)), z_star)[0, 1]
    if return_extra:
        return stat, None
    return stat


def fourier_transform(x):
    # fft (see formula between 17 and 18)
    z_fft = np.fft.fft(x) / len(x) ** 0.5
    # colect real and imag coeffs(see example 1, page 1013, 2nd paragraph)
    z_star = np.zeros(len(z_fft) * 2 - 1)
    z_star[0::2] = np.real(z_fft)
    z_star[1::2] = np.imag(z_fft[1:])
    return z_star[:len(z_fft)]


def legendre_projector(n):
    a = np.arange(n)
    basis = np.zeros((n, n))
    for k in a:
        basis[:, k] = (a - np.mean(a))**k
    q, _ = np.linalg.qr(basis)
    return q


def legendre_transform(x):
    n = len(x)
    q = legendre_projector(n)
    return x.dot(q)


def identity_transform(x):
    return x


def simulate_h0_distribution(n, d, transform, stat_fun, n_iter=200000, verbose=True, sim_verbose=False):
    cash_dir = '_fan98_temp'
    cash_file = os.path.join(cash_dir, 'h0_{}_{}_n{}_d{}_n_iter{}.npy'
                                       .format(stat_fun.__name__, transform.__name__, n, d, n_iter))
    if os.path.exists(cash_file):
        if verbose:
            print('Load from {}'.format(cash_file))
        stats_h0 = np.load(cash_file)
    else:
        if verbose or sim_verbose:
            print('Simulate and save to {}'.format(cash_file))
        stats_h0 = np.zeros(n_iter)
        for k in range(n_iter):
            if d is None:
                z_star = np.random.normal(size=n)
            else:
                z_star = np.random.standard_t(d, size=n)
            z_star = transform(z_star)
            stats_h0[k] = stat_fun(z_star, d)
        if not os.path.exists(cash_dir):
            os.makedirs(cash_dir)
        np.save(cash_file, stats_h0)
    return stats_h0


def get_p_val_one_tailed(val, h0_distribution):
    p_val = np.sum(val < h0_distribution)/h0_distribution.shape[0]
    return p_val


def get_p_val_two_tailed(val, h0_distribution):
    upper_count = np.sum(np.abs(val) < h0_distribution)
    lower_count = np.sum(h0_distribution < -np.abs(val))
    p_val = (upper_count + lower_count) / h0_distribution.shape[0]
    return p_val

if __name__ == '__main__':
    n = 20

    import pandas as pd
    paper_table = pd.read_csv('release/stats/upperquartile.csv', sep=';').values
    p_vals = paper_table[0, 1:]
    stats = paper_table[paper_table[:, 0] == n, 1:][0]


    d = None
    stats_h0 = simulate_h0_distribution(n, d, transform='legendre')


    levels = np.zeros_like(p_vals)
    for k, p_val in enumerate(p_vals):
        levels[k] = np.quantile(stats_h0, 1 - p_val)

    print(' '.join(['{:.2f}'.format(p*100) for p in p_vals]))
    print(' '.join(['{:.2f}'.format(level) for level in levels]))
    print(' '.join(['{:.2f}'.format(level) for level in stats]))
    print(' '.join(['{:.2f}'.format(level) for level in (stats-levels)/stats*100]))

