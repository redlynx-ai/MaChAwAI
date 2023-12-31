# -*- coding: utf-8 -*-
"""https://github.com/SFNGGL/machawai_containerized

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fqLe_THf277P-9oSTCAq7P7bBDgL9Qe4

Installazione pacchetti aggiuntivi
"""

"""## Inferenza con ONNX"""

"""Import"""

import onnx
import onnxruntime as onnxrt
import json
from machawai.ml.data import ITSDatasetConfig, parse_feature, InformedTimeSeries 
from machawai.labels import *
from machawai.ml.transformer import *
from machawai.adapters.pytorch import WrapTransformer
import io
import matplotlib.pyplot as plt
import pandas as pd

"""Check modello ONNX"""

# ONNX_PATH = f"code/model/gru_autoencoder.onnx"
ONNX_PATH = f"code/model/AD3_no_interp.onnx"

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

"""Read csv and json binary streams, code taken from
   the github repository"""
def read_bin_its(stream: tuple, config: ITSDatasetConfig):
  series = stream[0]
  json_data = stream[1]

  # returns a pandas.core.frame.DataFrame obj then passed on to ITS constructor
  # initialize one from data array with names
  
  """With the new implementation, parsing the dat file returns a DataFrame obj,
  properly truncated"""
  
  data_train=[]
  data_target=[]

  for col in series.columns:
    if config.isTrain(col):
      data_train.append(col)
    if config.isTarget(col):
      data_target.append(col)

  features = []
  features_train = []
  features_target = []

  ft_object = json.loads(json_data)
  for key in ft_object.keys():
    parsed = parse_feature(name=key, value=ft_object[key], config=config)
    features.append(parsed)
    if config.isTrain(key):
      features_train.append(key)
    if config.isTarget(key):
      features_target.append(key)

  its_name = "its_config"
  
  return InformedTimeSeries(
    series=series,
    features=features,
    data_train=data_train,
    data_target=data_target,
    features_train=features_train,
    features_target=features_target,
    name = its_name)

"""Funzione ausiliare per la lettura dei dati"""

def load_data(data:tuple, stat_path: str):
  # Configurazione di default
  config = ITSDatasetConfig.default()
  # Seleziona le colonne di input: Spostamento e Carico
  config.addTrainData(DISPLACEMENT)
  config.addTrainData(LOAD)
  # Seleziona le informazioni di input: Larghezza e Spessore
  config.addTrainFeature(WIDTH)
  config.addTrainFeature(THICKNESS)
  config.addTrainFeature("LS")
  config.addTrainFeature("LU")
  config.addTrainFeature("BP")
  # Seleziona la colonna di output: Estensometro
  config.addTargetFeature(EXTENSOMETER)
  # Leggi i dati della prova
  sample = read_bin_its(data, config)
  # Leggi il file di statistiche
  stat_df = pd.read_csv(stat_path, index_col = 0)
  print(sample.series)
  # Processamento:
  print(sample.series[LOAD])
  print(sample.series[DISPLACEMENT])
  # 1) Normalizzazione
  MinMaxSeriesNormalizer(df=stat_df, inplace=True).transform(sample)

  print("NORM")
  print(sample.series[LOAD])
  print(sample.series[DISPLACEMENT])

  # BSplineInterpolate(size=1000, inplace=True).transform(sample)

  print("INTERP")
  print(sample.series[LOAD])
  print(sample.series[DISPLACEMENT])

  # 2) Taglio fino al valore di massimo carico 
  # sample = CutSeriesToMaxIndex(colname=STRESS, inplace=False).transform(sample)
  # 3) Incapsulamento
  sample = WrapTransformer(add_batch_dim = True, device = "cpu").transform(sample)
  # Estrazione dati di input
  curve, info = sample.X()
  # Conversione in array numpy
  curve = curve.numpy()
  info = info.numpy()

  return curve, info, sample

def denormalize_data(its, exts_predicted, stat_path):
  print(exts_predicted)
  its.series["EXTS_PRE"] = exts_predicted
  stat_df = pd.read_csv(stat_path, index_col = 0)
  stat_df["EXTS_PRE"] = stat_df["EXTS_STRAIN"]
  print(its.getFeature('BP').value + 1)
  # BSplineInterpolate(size=its.getFeature('BP').value + 1, inplace=True).transform(its)
  MinMaxSeriesNormalizer(df=stat_df, inplace=True).revert(its)
  exts_predicted = its.series["EXTS_PRE"]
  print(exts_predicted)
  its.series.drop(["EXTS_PRE"], axis=1)

  return (its.series, exts_predicted)

def onnx_inf(data: tuple):

  STAT_PATH = f"code/data/d3_statistics.csv"

  curve, info, sample = load_data(data, stat_path = STAT_PATH)

  temp = info.copy()
  info[0, -2] = temp[0, -1]
  info[0, -1] = temp[0, -2]
  del temp

  print("Curve shape =", curve.shape)
  print("Info shape =", info.shape)

  onnx_session = onnxrt.InferenceSession(ONNX_PATH)
  onnx_inputs = {"disp_load_curve": curve, 
                "info": info}

  # dictionary = {"DISP": onnx_inputs['disp_load_curve'][0, :, 0].tolist(),
  #           "LOAD": onnx_inputs['disp_load_curve'][0, :, 1].tolist(),
  #           "info": onnx_inputs['info'][0, :].tolist()}

  # with open('/Users/ale/Desktop/onnx_inputs_ale.json', "w") as outfile:
  #     json.dump(dictionary, outfile)
  
  print(curve)
  print(info)

  onnx_output = onnx_session.run(["pred_exts"], onnx_inputs)

  print("Output shape =", onnx_output[0].shape)

  exts_predicted = onnx_output[0][0][:,0]

  results_denorm = denormalize_data(sample, exts_predicted, STAT_PATH)

  # print(f"Results are =\n{exts_predicted}")

  return results_denorm
  return (sample.series, exts_predicted)

  # return exts_predicted

if __name__ == '__main__':
  """Caricamento dati"""

  SAMPLE_PATH = f"./B01-XY-000-01.dat"
  STAT_PATH = f"./statistics.csv"
  curve, info, sample = load_data(sample_path = SAMPLE_PATH,
                                  stat_path = STAT_PATH)
  print("Curve shape =", curve.shape)
  print("Info shape =", info.shape)

  """Inferenza con ONNX"""

  onnx_session = onnxrt.InferenceSession(ONNX_PATH)
  onnx_inputs = {"disp_load_curve": curve, 
                "info": info}
  onnx_output = onnx_session.run(["pred_exts"], onnx_inputs)

  print("Output shape =", onnx_output[0].shape)

  """Visualizza risultati"""

  exts_predicted = onnx_output[0][0][:,0]

  # plt.plot(sample.getColumn(EXTENSOMETER), sample.getColumn(LOAD), label="real")
  # plt.plot(exts_predicted, sample.getColumn(LOAD), label="predicted")
  # plt.legend(loc="best")
  # plt.show()
