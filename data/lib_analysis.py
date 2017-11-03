import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_window(y, t, sr, wt=0.4):
    ws = wt * sr
    t0 = t - (wt / 2)
    t1 = t + (wt / 2)
    w0 = t0 * sr
    w1 = t1 * sr
    w0 = int(max(0, w0))
    w1 = int(min(y.size, w1))
    return y[w0:w1]

def window_energy(y, t, sr, wt=0.4):
    s = get_window(y, t, sr, wt)
    p = sp.sum(s * s)
    return p / s.size

def zero_cross_rate(y, t, sr, wt=0.4):
    s = get_window(y, t, sr, wt)
    return (np.diff(np.sign(s)) != 0).sum()

def plot_power_regress(y, sr, t_start=6, t_end=4, std_thresh=None, num_steps=1000, normalize=False):
    plt.clf()
    ts, ps = points_power(y, sr, t_start, t_end, std_thresh, num_steps) if not normalize else points_power_normalize(y, sr, t_start, t_end, std_thresh, num_steps)
    plt.plot(ts, ps)
    regress_function = power_regress_function(y, sr, t_start, t_end, std_thresh, normalize=normalize)
    plt.plot(ts, [regress_function(t) for t in ts])
    plt.xlabel("Time" + (" (s)" if not normalize else ""))
    plt.ylabel("Signal Power")
    plt.show()

def points_power(y, sr, t_start=6, t_end=4, std_thresh=None, num_steps=1000):
    ts0 = np.arange(t_start, y.size / sr - t_end, ((y.size / sr - t_end) - t_start) / num_steps)
    ps0 = [window_energy(y, t, sr) for t in ts0]
    ps_thresh = np.mean(ps0) + std_thresh * np.std(ps0) if std_thresh is not None else -np.inf
    ts = []
    ps = []
    for i in range(len(ts0)):
        if ps0[i] > ps_thresh:
            ts.append(ts0[i])
            ps.append(ps0[i])
    return ts, ps

def points_power_normalize(y, sr, t_start=6, t_end=4, std_thresh=None, num_steps=1000):
    start_time = t_start
    end_time = y.size / sr - t_end
    ts0 = np.arange(0, 1, (end_time - start_time) / num_steps)
    ps0 = [window_energy(y, start_time + t * (end_time - start_time), sr) for t in ts0]
    ps_thresh = np.mean(ps0) + std_thresh * np.std(ps0) if std_thresh is not None else -np.inf
    ts = []
    ps = []
    for i in range(len(ts0)):
        if ps0[i] > ps_thresh:
            ts.append(ts0[i])
            ps.append(ps0[i])
    return ts, ps


def power_mean(y, sr, t_start=6, t_end=4, std_thresh=None, num_steps=1000, normalize=False):
    ts, ps = points_power(y, sr, t_start, t_end, std_thresh, num_steps) if not normalize else points_power_normalize(y, sr, t_start, t_end, std_thresh, num_steps)
    return np.mean(np.array(ps))

def power_regress_function(y, sr, t_start=6, t_end=4, std_thresh=None, num_steps=1000, normalize=False):
    ts, ps = points_power(y, sr, t_start, t_end, std_thresh, num_steps) if not normalize else points_power_normalize(y, sr, t_start, t_end, std_thresh, num_steps)
    slope, intercept, r_val, p_val, err = stats.linregress(ts, ps)
    return lambda t : slope * t + intercept

    
def power_regress_slope(y, sr, t_start=6, t_end=4, std_thresh=None, num_steps=1000, normalize=False):
    ts, ps = points_power(y, sr, t_start, t_end, std_thresh, num_steps) if not normalize else points_power_normalize(y, sr, t_start, t_end, std_thresh, num_steps)
    slope, intercept, r_val, p_val, err = stats.linregress(ts, ps)
    return slope

def get_file_count(file_name, counts_file):
    c_file = open(counts_file, "r")
    c_lines = [l.replace("\n", "") for l in c_file.readlines() if "," in l]
    c_file.close()
    c_splits = [l.split(",") for l in c_lines]
    count = -1
    match_name = file_name if "/" not in file_name else file_name.split("/")[-1]
    for p in c_splits:
        f_name = p[0]
        if f_name == match_name:
            count = int(p[1])
            break
    return count

def get_file_sex(file_name, info_file):
    i_file = open(info_file, "r")
    i_lines = [l.replace("\n", "") for l in i_file.readlines() if "," in l]
    i_file.close()
    i_splits = [l.split(",") for l in i_lines]
    sex = None
    match_name = file_name if "/" not in file_name else file_name.split("/")[-1]
    for p in i_splits:
        f_name = p[0]
        if f_name == match_name:
            sex = p[1]
            break
    return sex

def get_file_age(file_name, info_file):
    i_file = open(info_file, "r")
    i_lines = [l.replace("\n", "") for l in i_file.readlines() if "," in l]
    i_file.close()
    i_splits = [l.split(",") for l in i_lines]
    age = -1 
    match_name = file_name if "/" not in file_name else file_name.split("/")[-1]
    for p in i_splits:
        f_name = p[0]
        if f_name == match_name:
            age = int(p[3])
            break
    return age 

