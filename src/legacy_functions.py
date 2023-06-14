R_cleave_relative_change = 0.05 # Relative change of R_cleave
R_cleave_steps_number = 100 # The number of changing R_cleave

peeks_relative_height = 0.07
signal_max_value = 5
signal_min_value = 0
points_in_cut_before_raise = 400  # The number of points to take in cut_data() function before raise !!!!
points_in_cut_after_raise = 300  # The number of points to take in cut_data() function after raise
points_in_cut_before_fall = 400  # The number of points to take in cut_data() function before fall  !!!
points_in_cut_after_fall = 300   # The number of points to take in cut_data() function after fall
r2_array = []
max_count = []
quad_approx_params = []
R_cleave_array = []

# def normalize_data():
#     amplification_factor = output_power / input_power / capler_coeff
#     new_amplitude = []
#     for i in range(len(amplitude)):
#         new_amplitude.append(
#             amplitude[i] /
#             oscillograph_param1 /
#             oscillograph_param2 /
#             input_power /
#             capler_coeff /
#             (amplification_factor ** 2)
#         )
#     return new_amplitude

# def find_maximums(function):
#     min = np.min(function)
#     R_eff_array = np.asarray(function)
#     # maximums = find_peaks(R_eff_array[points_to_smooth: len(R_eff_array) - points_to_smooth], height=0.00023, distance=800)
#     maximums = find_peaks(R_eff_array, height=min * (1 + peeks_relative_height), distance=800)
#     print(maximums)
#     print(len(maximums[0]))
#     maximums_x = []
#     maximums_y = []
#     result_file = open('maximums.txt', "w")
#     for i in range(len(maximums[0])):
#         maximums_x.append(maximums[0][i])
#         maximums_y.append(maximums[1]['peak_heights'][i])
#         result_file.write(str(maximums[0][i]) + '\t' + str(maximums[1]['peak_heights'][i]) + '\n')
#     result_file.close()
#     return maximums_x, maximums_y


# def get_R_eff(R_cleave):
#     new_R_eff = []  # "Максимумы" спектра отражения
#     print('R_cleave try is:', R_cleave)
#     for i in range(len(amplitude)):
#         if amplitude[i] < R_cleave:
#             new_R_eff.append(2 * R_cleave - amplitude[i])   # mean + (mean - R_eff)
#         else:
#             new_R_eff.append(amplitude[i])
#     return new_R_eff
#
#
# def get_sqrt():
#     new_amplitude = []
#     for i in range(len(amplitude)):
#         new_amplitude.append(np.sqrt(amplitude[i]))
#     return new_amplitude

# def find_R_cleave():
#     mean = np.mean(amplitude)
#     interval = mean * R_cleave_relative_change * 2 / R_cleave_steps_number
#     print(interval)
#     R_cleave_try = mean * (1.0 - R_cleave_relative_change)
#     new_R_eff = []
#     for i in range(R_cleave_steps_number): #range(R_cleave_steps_number):
#         new_R_eff = get_R_eff(R_cleave_try)
#         R_eff = new_R_eff
#         maximums_x, maximums_y = find_maximums(R_eff)
#         max_count.append(len(maximums_x))
#         r2, params = power_fit(maximums_x, maximums_y)
#         r2_array.append(r2)
#         quad_approx_params.append(params)
#         R_cleave_array.append(R_cleave_try)
#         R_cleave_try += interval
#     r2_max = np.max(r2_array)
#     print('r2_array = ', r2_array)
#     print('max_count = ', max_count)
#     print('r2_max = ', r2_max)
#
#     return r2_max,



# def write_for_gnuplot(start, end, filename, x, amplitude, signal, R_eff):
#     result_file = open(filename, "w")
#     for idx in range(start, end):
#         R_eff_string = '\n' if len(R_eff) == 0 else '\t' + str(R_eff[idx]) + '\n'
#         result_file.write(str(x[idx]) + '\t' + str(amplitude[idx]) + '\t' + str(signal[idx]) + R_eff_string)
#     result_file.close()
