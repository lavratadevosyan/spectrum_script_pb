import numpy as np

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth.tolist()


def linear_fit(x, signal, points_in_cut_after_fall):
    x_to_fit = x[points_in_cut_after_fall : int(len(x) / 2)]
    signal_to_fit = signal[points_in_cut_after_fall : int(len(x) / 2)]
    params = np.polyfit(x_to_fit, signal_to_fit, 1)
    return params


def power_fit(maximums_x, maximums_y, r2_score):
    params = np.polyfit(maximums_x, maximums_y, 4)
    p = np.poly1d(params)
    r2 = r2_score(maximums_y, p(maximums_x))
    return r2, params
