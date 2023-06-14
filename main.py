import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings

warnings.filterwarnings("ignore")

path = os.path.abspath(os.path.join(os.path.dirname(__file__), './src'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from processors import smooth
from fileutils import import_measurement_data, read_config, write_data_to_text_file
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks

from scipy.fftpack import fft

signal_relative_noise_level = 0.03
phase_relative_threshold = 0.1
points_to_smooth = 0

point_search_accuracy = 1
edge_point_bias = 0  # Сдвиг найденных ключевых точек на указанное число

oscillograph_param1 = 31
oscillograph_param2 = 1e6
input_power = 250  # мкВт
output_power = 5700  # мкВт
capler_coeff = 19

timeCoeff = 2.000000e-08  # Коэффициент оси х осциллографа

def reflect_array(array, reference):
    reflected = []
    for i in range(len(array)):
        if array[i] < reference:
            reflected.append(2 * reference - array[i])
        else:
            reflected.append(array[i])
    return reflected


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def find_edge_points(array, min, max, mid, type):
    points = []
    if type == 1:
        threshold = (max - min) * signal_relative_noise_level
    elif type == 2:
        threshold_up = (max - mid) * signal_relative_noise_level
        threshold_down = (mid - min) * signal_relative_noise_level
    else:
        print('Invalid type')
        return
    for i in range(1, (len(array) - point_search_accuracy) // point_search_accuracy):
        previous = array[(i - 1) * point_search_accuracy]
        next = array[(i + 1) * point_search_accuracy]
        real_current = array[i * point_search_accuracy]
        сurrent = -1
        if type == 1:
            if previous <= min + threshold and next > min + threshold:
                сurrent = i - edge_point_bias
            if next >= max - threshold and previous < max - threshold:
                сurrent = i + edge_point_bias
            if previous >= max - threshold and next < max - threshold:
                сurrent = i - edge_point_bias
            if next <= min + threshold and previous > min + threshold:
                сurrent = i + edge_point_bias
        elif type == 2:
            if previous >= mid - threshold_down and next < mid - threshold_down:
                сurrent = i - edge_point_bias
            if real_current < previous and real_current < next:
                сurrent = real_current
            if next >= mid - threshold_down and previous < mid - threshold_down:
                сurrent = i + edge_point_bias

            if previous <= mid + threshold_up and next > mid + threshold_up:
                сurrent = i - edge_point_bias
            if previous <= max - threshold_up and next < max - threshold_up and real_current > previous and real_current > next:
                сurrent = real_current
            if next <= mid + threshold_up and previous > mid + threshold_up:
                сurrent = i + edge_point_bias


        if сurrent > -1:
            if len(points) > 0:
                if abs(сurrent - points[-1] > point_search_accuracy):
                    points.append(сurrent)
            else:
                points.append(сurrent)
    if type == 2:
        reflected_signal = reflect_array(signal, mid_value)
        peaks = find_peaks(np.array(reflected_signal), distance=100000, height=(max_value - mid_value) / 2 + mid_value)
        points.extend(peaks[0])
        points.sort()
    return points

"""
@notice Определяет фазу сигнала с типом данных 1. Возвращает 0 при нулевом начальном уровне, 1 при максимальном начальном уровне,
-1 в случае ошибки

"""
def detect_phase(signal, min, max, threshold):
    if abs(signal[points_to_smooth + 1] - min) < threshold * (max - min):
        return 0
    if abs(signal[points_to_smooth + 1] - max) < threshold * (max - min):
        return 1
    return -1

"""
@notice Определяет фазу сигнала с типом данных 2. Возвращает 0 при нулевом начальном уровне, 1 при максимальном начальном уровне,
-1 в случае ошибки

"""
def detect_phase_type_2(signal, mid, min, max, threshold, points):
    if signal[points[2]] - mid < threshold * (min - mid):
        return 0
    if signal[points[2]] - mid > threshold * (max - mid):
        return 1
    return -1


"""
@notice Находит средние точки амплитуды на указанных промежутках
@param amplitude - Массив значений амплитуды
@param points - Массив точек
@param start_point - Начальная точка в массиве точек, показывающая период, с которого начинать считать
@param rel_start_interval_point - Начальная точка интервала, выраженная в % от всей длины интервала
@param rel_interval_size - Длина интервала для среднего, выраженная в % от всей длины интервала
"""
def find_average_values(amplitude, points, start_point, rel_start_interval_point, rel_interval_size, type):
    average_values = []
    average_values_x = []
    current_point = start_point
    while current_point + 1 < len(points) - 1:
        full_interval = points[current_point + 1] - points[current_point]
        start_interval_point = full_interval * rel_start_interval_point // 100 + points[current_point]
        end_interval_point = full_interval * rel_interval_size // 100 + start_interval_point
        average_values.append(
            np.average(
                amplitude[
                    start_interval_point:
                    end_interval_point
                ]
            )
        )
        average_values_x.append(
            (start_interval_point + end_interval_point) / 2
        )
        if type == 1:
            current_point += 2
        elif type == 2:
            current_point += 3
    return average_values_x, average_values


def cut_data(signal, amplitude, points, peak_number, phase, toReflect, type):
    if type == 1:
        if phase == 1:
            first_point = 3 + 4 * (peak_number - 1)
        else:
            first_point = 4 * (peak_number - 1)
        subpoints = points[first_point:first_point + 4]

    elif type == 2:
        if phase == 0:
            first_point = 1 + 3 + 6 * (peak_number - 1)
        else:
            first_point = 1 + 6 * (peak_number - 1)
        subpoints = points[first_point:first_point + 5]

    if type == 1:
        if toReflect:
            new_signal = signal[subpoints[3]:subpoints[2]:-1]   # inversed decline
            new_amplitude = amplitude[subpoints[3]:subpoints[2]:-1] + amplitude[subpoints[0]:subpoints[1]:1] # inversed decline + incline
            bias = new_signal[-1]
            for i in range(subpoints[0], subpoints[1]):
                new_signal.append(signal[i] + bias)
        else:
            new_signal = signal[subpoints[0]:subpoints[1]] + signal[subpoints[2]:subpoints[3]]
            new_amplitude = amplitude[subpoints[0]:subpoints[1]] + amplitude[subpoints[2]:subpoints[3]]
        new_x = list(range(subpoints[2] - subpoints[3], 0)) + list(range(0, subpoints[1] - subpoints[0]))

    elif type == 2:
        if toReflect:
            new_signal = signal[subpoints[4]:subpoints[3]:-1] + signal[subpoints[0]:subpoints[1]] # # inversed decline
            new_amplitude = amplitude[subpoints[4]:subpoints[3]:-1] + amplitude[subpoints[0]:subpoints[1]:1] # inversed decline + incline
        else:
            new_signal = []
            bias = signal[subpoints[0]]
            for i in range(subpoints[0], subpoints[1]):
                new_signal.append(signal[i] - bias)
            new_signal = new_signal + signal[subpoints[3]:subpoints[4]]
            new_amplitude = amplitude[subpoints[0]:subpoints[1]] + amplitude[subpoints[3]:subpoints[4]]
        new_x = list(range(subpoints[3] - subpoints[4], 0)) + list(range(0, subpoints[1] - subpoints[0]))

    return new_x, new_signal, new_amplitude, subpoints


print("Reading config data...")
config = read_config("./config.json")
print("Done.")
edge_point_bias = config['EDGE_POINT_BIAS']
points_to_smooth = config['POINTS_TO_SMOOTH']
if config['TYPE'] > 2 or config['TYPE'] < 1:
    raise Exception("Incorrect type")
for file_index in range(len(config['INPUT_FILES'])):
    print("Importing data from", config['INPUT_FILES'][file_index], "files...")
    x, signal, amplitude, amplitude_2 = import_measurement_data('./data/' + config['INPUT_FILES'][file_index], 'csv', 2)
    max_value = np.amax(signal)
    min_value = np.amin(signal)
    mid_value = np.average(signal)
    print("Done.", "Max signal value:", max_value, "Min signal value:", min_value, "Average signal value:", mid_value)
    print("Smoothing...")
    amplitude = smooth(amplitude, points_to_smooth)
    amplitude_2 = smooth(amplitude_2, points_to_smooth)
    signal = smooth(signal, points_to_smooth)
    print("Done.")


    print("Detecting edge points...")
    points = find_edge_points(
        signal,
        min_value,
        max_value,
        mid_value,
        config["TYPE"]
    )
    print("Done.", len(points), "points found")
    point_values = []
    for i in range(len(points)):
        point_values.append(signal[points[i]])


    print("Detecting phase...")
    if config["TYPE"] == 1 :
        phase = detect_phase(signal, min_value, max_value, phase_relative_threshold)
    elif config["TYPE"] == 2:
        phase = detect_phase_type_2(signal, mid_value, min_value, max_value, phase_relative_threshold, points)
    if phase == -1:
        print("ERROR: Unable to detect phase, skipping...")
        continue
    print("Done.", phase, "phase detected")

    print('Searching for average values...')
    start_point = 0
    if config["TYPE"] == 1:
        start_point = 2 if phase == 1 else 1
    else:
        start_point = 3

    average_values_x, average_values = find_average_values(
        amplitude,
        points,
        start_point,
        90,
        10,
        config["TYPE"]
    )
    print('Done. Found', len(average_values), 'average values')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        x[:],
        signal[:],
        'b-',
        points[:],
        point_values[:],
        'bo',
        zorder=10
    )
    ax2.set_zorder(10)


    ax2.plot(
        x[points_to_smooth:-points_to_smooth],
        amplitude[points_to_smooth:-points_to_smooth],
        'g-',
        average_values_x,
        average_values,
        'ro',
        linewidth=0.5
    )
    ax2.set_zorder(5)

    max_average_point = np.amax(average_values)
    min_average_point = np.amin(average_values)
    extra_space_range = (max_average_point - min_average_point) * 0.1

    plt.ylim([min_average_point - extra_space_range, max_average_point + extra_space_range])

    ax1.grid(axis='both')
    plt.show()

    while True:
        peak_number = input("Enter peak number: ")
        if peak_number.isdigit():
            break
        else:
            print("Incorrect peak number")

    x_cut, signal_cut_reflected, amplitude_cut, subpoints = cut_data(signal, amplitude, points, int(peak_number), phase, True, config["TYPE"])
    _, signal_cut, amplitude_2_cut, _ = cut_data(signal, amplitude_2, points, int(peak_number), phase, False, config["TYPE"])

    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax01 = ax0.twinx()
    ax11 = ax1.twinx()
    ax0.plot(x_cut, signal_cut_reflected, 'r-')
    ax01.plot(x_cut, amplitude_cut, 'b-', linewidth=0.5)
    ax1.plot(x_cut, signal_cut, 'r-')
    ax11.plot(x_cut, amplitude_2_cut, 'b-', linewidth=0.5)

    ax1.grid(axis='both')
    ax2.grid(axis='both')
    plt.show()



    print('Sinus fitting...')
    sin_signal_1 = amplitude_2[subpoints[0]:subpoints[1]]
    end_point = subpoints[3] if config['TYPE'] == 1 else subpoints[4]
    start_point = subpoints[2] if config['TYPE'] == 1 else subpoints[3]
    sin_signal_2 = amplitude_2[start_point:end_point]
    x_sin_signal_1 = np.array(x[subpoints[0]:subpoints[1]])
    x_sin_signal_2 = np.array(x[start_point:end_point])

    res1 = fit_sin(x_sin_signal_1, sin_signal_1)
    res2 = fit_sin(x_sin_signal_2, sin_signal_2)
    # print("Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res)


    print("Getting FFT...")

    N1 = len(x_sin_signal_1)
    T1 = 1.0/1000.0
    y1f = fft(sin_signal_1)
    x1f = np.linspace(0.0, 1.0 / (2.0 * T1), N1 // 2)

    N2 = len(x_sin_signal_2)
    T2 = 1.0/1000.0
    y2f = fft(sin_signal_2)
    x2f = np.linspace(0.0, 1.0 / (2.0 * T2), N2 // 2)


    print("Done.")




    fig, (axs0, axs1) = plt.subplots(2, 2)
    fig.tight_layout(pad=2.0)
    axs0[0].title.set_text("phase=%(phase)s" % res1)
    axs0[0].plot(
        x_sin_signal_1,
        sin_signal_1,
        'r-',
        x_sin_signal_1,
        res1["fitfunc"](x_sin_signal_1),
        'b--'
    )
    axs0[1].title.set_text("phase=%(phase)s" % res2)
    axs0[1].plot(
        x_sin_signal_2,
        sin_signal_2,
        'r-',
        x_sin_signal_2,
        res2["fitfunc"](x_sin_signal_2),
        'b--'
    )

    axs1[0].plot(x1f, 2.0 / N1 * np.abs(y1f[:N1 // 2]))
    axs1[1].plot(x2f, 2.0 / N2 * np.abs(y2f[:N2 // 2]))

    plt.show()

    print("Done.")


    print("Writing data to", config['OUTPUT_FILES'][file_index], "text file...")
    write_data_to_text_file(
        './out/' + config['OUTPUT_FILES'][file_index],
        [
            'X',
            'SIGNAL',
            'AMP1',
            'AMP2'
        ],
        [
            x_cut,
            signal_cut,
            amplitude_cut,
            amplitude_2_cut
        ],
        '\t'

    )
    print("Done.")