# re.match
# pattern =
# '| *DATA *|'
# then pull next 6 lines and then data starts:

import os, re, io, sys
import pandas as pd
import numpy as np
from break_point import *
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen

PATTERN_1 = r"\W\s+(\w+)(\s*)(\w+)(\s*)(\w+)(\s*)(\w+)(\s*)(\w+)(\s*)\s+\W\n"
PATTERN_2 = r"\W\s+(D\w+)\s+\W\n" # find DATA lines
PATTERN_3 = r"\W\s+(\w+)\s+\W\n"
PATTERN_4 = r"\W\s+(\w+)(\s*)\s+\W\n" 
PATTERN_5 = r"\W\s+(\w+)(\s+)(\w+)(\s+)(\w+)(\s+)(\w+)(\s+)(\w+)(\s+)(\w+)(\s+)\s+\W\n"
# PATTERN_BRK = r""

def find_row_number(file_path : str) -> tuple:
    if os.path.isfile(file_path):
        matches = [-1, -1]
        row1, row2 = 0, 0
        with open(f'{file_path}', 'r+') as file:
            for line in file:
                # z = re.find(PATTERN_2, row)
                z1 = re.match(PATTERN_2, line)
                if z1:
                    matches[0] = row1
                z2 = re.match(PATTERN_5, line)
                if z2:
                    print(z2)
                    matches[1] = row2
                row1 += 1
                row2 += 1
            file.close()
            return tuple(matches)

def extract_data(file_path: str) -> pd.DataFrame:
    if os.path.isfile(file_path):
        data_array = []
        with open(f"{file_path}", 'r') as file:
            for line in file:
                z = re.match(PATTERN_2, line)
                if z:
                    skip_n_lines(file, 6)
                    for row_of_data in file:
                        row = row_of_data.split(';')
                        if len(row) > 3:
                            data_array.append( ( row[4], row[2], row[3] ) )
                        else:
                            break
                    file.close()
                    # return np.asarray(data_array, dtype = 'float64')
                    data_array = np.asarray(data_array, dtype = 'float64')
                    print("data_array:", data_array.shape)
                    brkpnt = show_break_point(data_array)
                    print("brkpnt:", brkpnt)
                    data_array = data_array[:brkpnt[0]+1]
                    print("data_array_brkpnt:", data_array.shape)
                    return pd.DataFrame( data_array, columns = ["DISP","LOAD","EXTS"] )
    return pd.DataFrame( np.zeros((1,3), dtype = 'float64') )

def extract_text(file_path: str) -> str:
    if os.path.isfile(file_path):
        text_array = ''
        with open(f"{file_path}", 'r') as file:
            for line in file:
                z = re.match(PATTERN_2, line) # matches the DATA line
                if z:
                    skip_n_lines(file, 2)
                    line = file.readline().split(';')
                    text_array += f"{line[2]},{line[1]},{line[3]}\n"
                    skip_n_lines(file, 4)
                    for row_of_data in file:
                        row = row_of_data.split(';')
                        if len(row) > 3:
                            text_array += f"{row[2]},{row[1]},{row[3]}\n"
                        else:
                            break
                    file.close()
                    return text_array
    return ""

def format_data(data_txt: str, data_pd: pd.DataFrame) -> str:
    brkpnt = show_break_point(data_pd.values)
    return data_txt[:brkpnt[1][0]]

def skip_n_lines(file: io.TextIOWrapper, n: int) -> None:
    for i in range(n):
        file.readline()
    # return file

def show_break_point(data: np.ndarray):
    assert data.shape[1] == 3
    # print(data.shape)
    return trova_rottura(data[:, 0], data[:, 1])

def read_dat(file_path: str) -> str:
    result = extract_data(file_path)
    text_values = extract_text(file_path)
    fin_array = format_data(text_values, result)
    return fin_array

def nuovo_path_file(file_path):
    # Ottieni il nome del file dal percorso fornito
    nome_file = os.path.basename(file_path)

    # Get the current script's directory
    if getattr(sys, 'frozen', False):
        directory_corrente = os.path.dirname(sys.executable)
    else:
        directory_corrente = os.path.dirname(os.path.realpath(__file__))

    # Ottieni la directory corrente dove risiede lo script
    # directory_corrente = os.path.dirname(os.path.realpath(__file__))

    # Crea il percorso completo per la cartella "results"
    cartella_results = os.path.join(directory_corrente, 'results')

    # Crea la cartella "results" se non esiste già
    if not os.path.exists(cartella_results):
        os.makedirs(cartella_results)
        print(f"Cartella 'results' creata.")
    else:
        print(f"Cartella 'results' esiste già.")

    # Crea il nuovo percorso del file
    nuovo_file_path = os.path.join(cartella_results, nome_file)

    return nuovo_file_path

def replace(file_path, features, pred_values):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            offset = ["","","","",""]
            data_section = False
            i = 0
            for line in old_file:
                if re.match(PATTERN_2, line):
                    data_section = True
                if data_section:
                    if len(features) > 0:
                        new_file.write(f"{features.pop(0)}\n")
                    elif len(offset) > 0:
                        new_file.write(f"{offset.pop(0)}{line}")
                    elif len(pred_values) > 0:
                        new_file.write(f"{line.strip()};{pred_values.pop(0)}\n")
                    else:
                        data_section = False
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    #remove(file_path)
    result_path = nuovo_path_file(file_path)
    #Move new file
    # move(abs_path, file_path)
    move(abs_path, result_path)

def usage():
    result = extract_data("B01-XY-000-01.dat")
    breakpoint = show_break_point(result.values)
    text = extract_text("B01-XY-000-01.dat")
    # print("Breaking point: ",breakpoint[0])
    # print("Section: ",breakpoint[1])
    return result, breakpoint, text

if __name__ == "__main__":
    print(read_dat(sys.argv[1]))

