import csv
import json


"""
@notice Импортирует все значения x, signal, amplitude и amplitude из файлов 'filename.format', 'filename_2.format' и 
'filename_3.format', начиная с 'dataStartsFrom' строки
"""
def import_measurement_data(filename, format, dataStartsFrom):
    new_x = []
    new_signal = []
    new_amplitude = []
    new_amplitude_2 = []
    files = [
        filename + "." + format,
        filename + "_2." + format,
        filename + "_3." + format
    ]
    for file in files:
        with open(file, newline='') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx < dataStartsFrom:    # Пропускаем шапку
                    continue
                if file == files[0]:
                    new_x.append(int(row[0]))
                    new_signal.append(float(row[1]))
                elif file == files[1]:
                    new_amplitude.append(float(row[1]))
                elif file == files[2]:
                    new_amplitude_2.append(float(row[1]))
                else:
                    print("IMPORT ERROR")
    return new_x, new_signal, new_amplitude, new_amplitude_2

"""
@notice Читает конфиг файл в формате .json
"""
def read_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

"""
@notice Записывает переданные массивы данных в файл
@param file_path - Относительный путь до файла
@param headers - Массив заголовков для каждого из столбцов
@param data_arrays - Массив массивов данных
@param separator - Разделитель
"""
def write_data_to_text_file(file_path, headers, data_arrays, separator):
    result_file = open(file_path, "w")
    headers_string = ""
    for header_idx in range(len(headers)):
        headers_string += str(headers[header_idx]) + separator
    result_file.write(headers_string + '\n')
    for string_idx in range(len(data_arrays[0])):
        output = ""
        for raw_idx in range(len(data_arrays)):
            output = output + str(data_arrays[raw_idx][string_idx]) + separator
        result_file.write(output + '\n')
    result_file.close()