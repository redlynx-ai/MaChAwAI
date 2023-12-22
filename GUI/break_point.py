import numpy as np

def derivata_seconda(X: 'list | np.ndarray', Y: 'list | np.ndarray') -> np.ndarray:
    """
        Calcola la derivata seconda di una funzione date 
        le sequenze di coordinate `X` ed `Y`.

            dY[i] = (Y[i+2] - 2*Y[i+1] + Y[i]) / (X[i+1] - X[i])^2 

        ATTENZIONE: `len(dY) = len(Y) - 2`.
    """
    f_len = len(Y)
    # la funzione deve essere definita in almeno 3 punti
    assert f_len >= 3
    der_len = f_len - 2
    # X ed Y devono avere la stessa lunghezza
    assert len(X) == f_len
    # inizializza la derivata come una sequenza di 0
    dY = [0.] * der_len
    for i in range(der_len):
        num = (Y[i+2] - 2*Y[i+1] + Y[i])
        den = (X[i+1] - X[i])**2
        if den == 0:
            dY[i] = np.Inf if num >= 0 else - np.Inf
        else:
            dY[i] = (Y[i+2] - 2*Y[i+1] + Y[i]) / (X[i+1] - X[i])**2
    return dY

def trova_sezione(Y: 'list | np.ndarray', p: float):
    """
        Calcola gli indici della sezione di interesse che 
        va da `Ymax` = massimo di Y fino ad `Ymax - p * Ymax`
    """
    # Calcola il punto iniziale della sezione come il massimo di Y
    start_point = np.argmax(Y)
    # Calcola il valore massimo di Y
    Ymax = np.max(Y)
    # Calcola il punto finale della sezione come Ymax - p% di Ymax
    Yp = Ymax - (p * Ymax)
    end_point = start_point
    for i in range(start_point + 1, len(Y)):
        if Y[i] < Yp:
            return start_point, end_point
        else:
            end_point = i
    return start_point, end_point

def max_var_point(der: 'list | np.ndarray'):
    """
        Restituisce il punto di massima variazione di una serie.
    """
    if len(der) <= 2:
        return 0
    max_var = der[1] - der[0]
    max_var_idx = 0
    for i in range(1, len(der) - 1):
        var = abs(der[i+1] - der[i])
        if var > max_var:
            max_var = var
            max_var_idx = i
    return max_var_idx

def trova_rottura(X: 'list | np.ndarray', Y: 'list | np.ndarray', p: float = 0.1):
    """
        Calcola l'indice del punto di rottura.
    """
    # Calcola derivata seconda dell'intera curva
    der2 = derivata_seconda(Y = Y, X = X)
    # Calcola la sezione di interesse
    start_section, end_section = trova_sezione(Y=Y, p=p)
    section_len = (end_section + 1) - start_section
    # Se la sezione di interesse è formata da 1 solo punto, quello è il punto di rottura
    if section_len <= 1:
        return start_section, (start_section, end_section)
    # Seleziona il tratto di derivata nella sezione di interesse
    der2_section = der2[start_section: end_section + 1]
    assert len(der2_section) + 2 >= end_section + 1 - start_section
    # Calcola il punto di massima variazione della derivata nella sezione di interesse
    break_point = max_var_point(der2_section)

    return start_section + break_point, (start_section, end_section)
