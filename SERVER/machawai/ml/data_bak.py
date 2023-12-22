"""
Classes and functions for ML data management.
"""

# --------------
# --- IMPORT ---
# --------------
import pandas as pd
import numpy as np
import os
import json
import bitarray
import copy
import random

# ------------------
# --- EXCEPTIONS ---
# ------------------

class ConfigLoadingError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ParsingError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)

# ----------------
# --- FEATURES ---
# ----------------

class Feature():

    def __init__(self,
                 name: str,
                 value) -> None:
        self.name = name
        self.value = value

    def isCategorical(self) -> bool:
        raise NotImplementedError()

    def encode(self) -> np.ndarray:
        raise NotImplementedError()
    
    def copy(self) -> 'Feature':
        return Feature(name=self.name, value=self.value)
    
    def dictionary(self) -> dict:
        return {"name": self.name,
                "value": self.value}
    
    def __repr__(self) -> str:
        return "(name: {}, value: {})".format(self.name, self.value)
    
    def __str__(self) -> str:
        self.__repr__()
    
class BooleanFeature(Feature):

    def __init__(self, name: str, value: 'bool') -> None:
        assert isinstance(value, bool)
        super().__init__(name, value)

    def isCategorical(self):
        return True

    def encode(self) -> np.ndarray:
        return np.array([1.0 if self.value else 0.0], dtype=float)
    
    def copy(self) -> 'BooleanFeature':
        return BooleanFeature(name=self.name, value=self.value)

class NumericFeature(Feature):

    def __init__(self, name: str, value: 'int | float') -> None:
        assert isinstance(value, int) or isinstance(value, float)
        super().__init__(name, value)

    def isCategorical(self) -> bool:
        return False

    def encode(self) -> np.ndarray:
        return np.array([self.value], dtype=float)
    
    def copy(self) -> 'NumericFeature':
        return NumericFeature(name=self.name, value=self.value)
    
class CategoricalFeature(Feature):

    def __init__(self, name: str, value: str, known_values: 'list[str]' = [], one_hot: bool = False) -> None:
        assert isinstance(value, str)
        super().__init__(name, value)
        self.known_values = known_values
        self.one_hot = one_hot

    def isCategorical(self) -> bool:
        return True

    def encode(self) -> np.ndarray:
        if self.one_hot:
            enc = [0.0] * len(self.known_values)
            enc[self.known_values.index(self.value)] = 1.0
            return np.array(enc, dtype=float)
        else:
            if len(self.known_values) > 0:
                return np.array([self.known_values.index(self.value)], dtype=float)
            else:
                # TODO: testare
                ba = bitarray.bitarray()
                enc = ba.frombytes(self.value.encode("utf-8")).tolist()
                return np.array(enc, dtype=float)
    
    def copy(self) -> 'CategoricalFeature':
        return CategoricalFeature(name=self.name, 
                                  value=self.value, 
                                  known_values=self.known_values, 
                                  one_hot=self.one_hot)

class VectorFeature(Feature):

    def __init__(self, name: str, value: list) -> None:
        assert isinstance(value, list)
        super().__init__(name, value)
    
    def isCategorical(self) -> bool:
        return False

    def encode(self) -> np.ndarray:
        return np.array(self.value, dtype=float)
    
    def copy(self) -> 'VectorFeature':
        return VectorFeature(name=self.name, value=copy.deepcopy(self.value))

class SeriesFeature(Feature):

    def __init__(self, value: pd.Series, name: str = None) -> None:
        if name == None:
            name = value.name
        super().__init__(name, value)

    def isCategorical(self) -> bool:
        return False

    def encode(self) -> np.ndarray:
        return self.value.values.astype(float)
    
    def copy(self) -> 'SeriesFeature':
        return SeriesFeature(value=self.value.copy(), name = self.name)

# ----------------------------
# --- INFORMED TIME SERIES ---
# ----------------------------

class InformedTimeSeries():

    def __init__(self,
                 series: pd.DataFrame,
                 features: 'list[Feature]',
                 data_train: 'list[str]' = [],
                 data_target: 'list[str]' = [],
                 features_train: 'list[str]' = [],
                 features_target: 'list[str]' = [],
                 name: str = None) -> None:
        self.series = series
        self.features = features
        self.data_train = data_train
        self.data_target = data_target
        self.features_train = features_train
        self.features_target = features_target
        self.name = name

    def getColnames(self):
        """
        Return the series column names
        """
        return self.series.columns
    
    def hasColumn(self, name: str):
        return name in self.getColnames()

    def getColumn(self, name:str):
        """
        Return the series column with the given name.
        """
        if self.hasColumn(name):
            return self.series.loc[:,name]
        raise ValueError("Unknown column with name {}".format(name))

    def getFeature(self, name:str):
        """
        Return the feature with the given name.
        """
        flist = list(filter(lambda f: f.name == name, self.features))
        if len(flist) == 0:
            raise ValueError("Unknown feature with name {}".format(name))
        return flist[0]
    
    def hasFeature(self, name:str):
        """
        Check if a feature with the give name exists.
        """
        return len(list(filter(lambda f: f.name == name, self.features))) > 0
    
    def dropFeature(self, name:str):
        """
        Drop the feature with the given name. 
        """
        self.features.pop(self.getFeature(name=name))

    def addFeature(self, feature: Feature):
        """
        Add a new feature.
        """
        if self.hasFeature(name=feature.name):
            raise ValueError("Feature with name {} still exists.".format(feature.name))
        self.features.append(feature)

    def hasTraining(self, name: str):
        """
        Check if a training feature with the give name exists.
        """
        return (self.hasFeature(name) and name in self.features_train) or \
               (self.hasColumn(name) and name in self.data_train)

    def getTraining(self, name: str):
        """
        Return the training feature or data column with the given name.
        """
        if self.hasTraining(name):
            if name in self.data_train:
                return self.series.loc[:,name]
            else:
                flist = list(filter(lambda f: f.name == name, self.features))
                return flist[0]
        raise ValueError("Unknown training feature or data column with name {}".format(name))

    def untrain(self, name: str):
        """
        Set the feature/data column with the given name as a non-training feature/data column. 
        """
        if name in self.data_train:
            self.data_train.remove(name)
        elif name in self.features_train:
            self.features_train.remove(name)

    def setTraining(self, name: str):
        """
        Set the feature/data column with the given name as a training feature/data column.
        """
        if not self.hasTraining(name):
            if self.hasColumn(name):
                self.data_train.append(name)
            elif self.hasFeature(name):
                self.features_train.append(name)
            else:
                raise ValueError("Unknown feature or data column with name {}".format(name))

    def hasTarget(self, name:str):
        """
        Check if a target feature/data column with the give name exists.
        """
        return (self.hasFeature(name) and name in self.features_target) or \
               (self.hasColumn(name) and name in self.data_target)

    def getTarget(self, name:str):
        """
        Return the target feature/data column with the given name.
        """
        if self.hasTarget(name):
            if name in self.data_target:
                return self.series.loc[:,name]
            else:
                flist = list(filter(lambda f: f.name == name, self.features))
                return flist[0]
        raise ValueError("Unknown target with name {}".format(name))
    
    def untarget(self, name:str):
        """
        Untarget the feature/data column with the given name. 
        """
        if name in self.data_target:
            self.data_target.remove(name)
        elif name in self.features_target:
            self.features_target.remove(name)

    def setTarget(self, name: str):
        """
        Set the feature/data column with the given name as target
        """
        if not self.hasTarget(name):
            if self.hasColumn(name):
                self.data_target.append(name)
            elif self.hasFeature(name):
                self.features_target.append(name)
            else:
                raise ValueError("Unknown feature or data column with name {}".format(name))
    
    def copySeries(self) -> pd.DataFrame:
        return self.series.copy()

    def copyFeatures(self) -> 'list[Feature]':
        f = []
        for feat in self.features:
            f.append(feat.copy())
        return f

    def getTrainSeries(self) -> pd.DataFrame:
        return self.series.loc[:,self.data_train]
    
    def getTargetSeries(self) -> 'pd.Series | pd.DataFrame':
        return self.series.loc[:,self.data_target]
    
    def getTrainFeatures(self) -> 'list[Feature]':
        return list(filter(lambda f: f.name in self.features_train, self.features))
    
    def getTargetFeatures(self) -> 'list[Feature]':
        return list(filter(lambda f: f.name in self.features_target, self.features))

    def copy(self) -> 'InformedTimeSeries':
        return InformedTimeSeries(series=self.copySeries(),
                                  features=self.copyFeatures(),
                                  data_train=self.data_train,
                                  data_target=self.data_target,
                                  features_train=self.features_train,
                                  features_target=self.features_target,
                                  name=self.name)

class ITSDatasetConfig():
    # TODO: creare le classi FeatureOptions
    def __init__(self,
                 data_file_name: str = "data",
                 features_file_name: str = "features",
                 data_separator: str = ",",
                 data_header: 'int | list[int]' = 0,
                 data_colnames: 'list[str]' = None,
                 data_train: 'list[str]' = [],
                 data_target: 'list[str]' = [],
                 features_options: dict = {},
                 features_train: 'list[str]' = [],
                 features_target: 'list[str]' = []) -> None:
        self.data_file_name = data_file_name
        self.features_file_name = features_file_name
        self.data_separator = data_separator
        self.data_header = data_header
        self.data_colnames = data_colnames
        self.data_train = data_train
        self.data_target = data_target
        self.data_type = float
        self.features_options = features_options
        self.features_train = features_train
        self.features_target = features_target

    def setColnames(self, colnames: 'list[str]'):
        self.data_colnames = colnames

    def addTrainData(self, name: str):
        if not name in self.data_train:
            self.data_train.append(name)

    def addTargetData(self, name: str):
        if not name in self.data_target:
            self.data_target.append(name)

    def addTrainFeature(self, name: str):
        if not name in self.features_train:
            self.features_train.append(name)

    def addTargetFeature(self, name: str):
        if not name in self.features_target:
            self.features_target.append(name)

    def isTrain(self, name: str) -> bool:
        """
        Check if a given feature or series column is a train feature.
        """
        return (name in self.data_train) or (name in self.features_train)

    def isTarget(self, name:str) -> bool:
        """
        Check if a given feature or series column is a target feature.
        """
        return (name in self.data_target) or (name in self.features_target)
        
    def knownValues(self, name:str) -> 'list[str]':
        """
        Return the known values for a given feature.
        """
        if name in self.features_options.keys():
            if "values" in self.features_options[name]:
                return self.features_options[name]["values"]
        return []
    
    def oneHot(self, name:str) -> bool:
        """
        Return the one_hot option value for a given feature.
        """
        if name in self.features_options.keys():
            if "one_hot" in self.features_options[name]: 
                return self.features_options[name]["one_hot"]
        return False

    def default():
        return ITSDatasetConfig()
    
    def fromDictionary(config: dict):
        """
        Load configuration from dictionary.
        """
        itsd_config = ITSDatasetConfig.default()
        # Read data configurations
        if "data" in config.keys():
            data_config = config["data"]
            if "file_name" in data_config.keys():
                itsd_config.data_file_name = data_config["file_name"]
            if "separator" in data_config.keys():
                itsd_config.data_separator = data_config["separator"]
            if "header" in data_config.keys():
                itsd_config.data_header = data_config["header"]
            if "colnames" in data_config.keys():
                itsd_config.data_colnames = data_config["colnames"]
            if "train" in data_config.keys():
                itsd_config.data_train = data_config["train"]
            if "target" in data_config.keys():
                itsd_config.data_target = data_config["target"]
        # Read features configurations
        if "features" in config.keys():
            features_config = config["features"]
            if "file_name" in features_config.keys():
                itsd_config.features_file_name = features_config["file_name"]
            if "train" in features_config.keys():
                itsd_config.features_train = features_config["train"]
            if "target" in features_config.keys():
                itsd_config.features_target = features_config["target"] 
            if "options" in features_config.keys():
                itsd_config.features_options = features_config["options"]           
        return itsd_config

    def fromFile(file: str):
        """
        Load configuration from json.
        """
        try:
            with open(file, "r") as config_file:
                config = json.load(config_file)
                return ITSDatasetConfig.fromDictionary(config)
        except:
            raise ConfigLoadingError()
    
    def dictionary(self) -> dict:
        return {
                "data": {
                    "file_name": self.data_file_name,
                    "separator": self.data_separator,
                    "header": self.data_header,
                    "colnames": self.data_colnames,
                    "train": self.data_train,
                    "target": self.data_target
                },
                "features": {
                    "filename": self.features_file_name,
                    "train": self.features_train,
                    "target": self.features_target,
                    "options": self.features_options
                }
            }

    def __repr__(self) -> str:
        return self.dictionary().__repr__()

    def __str__(self) -> str:
        return self.__repr__()

class InformedTimeSeriesDataset():

    def __init__(self,
                 data: 'list[InformedTimeSeries]',
                 config: 'ITSDatasetConfig',
                 transformers: 'list' = []) -> None:
        self.data = data
        self.config = config
        self.transformers = transformers

    def size(self) -> int:
        return len(self.data)

    def shuffle(self):
        random.shuffle(self.data)

    def train_test_split(self, test_ratio: float = 0.1, shuffle: bool = False, random_seed: int = None):
        """
        Split the dataset into train and test subsets.
        """
        assert test_ratio <= 1, test_ratio >= 0.
        test_size = int(self.size() * test_ratio)
        train_size = self.size() - test_size

        indices = np.arange(self.size(), dtype=int)
        if shuffle:
            random.seed(random_seed)
            random.shuffle(indices)
        train_indices = indices[:train_size].tolist()
        test_indices = indices[train_size:].tolist()

        return InformedTimeSeriesDataset([self.data[i] for i in train_indices], self.config, self.transformers), \
               InformedTimeSeriesDataset([self.data[i] for i in test_indices], self.config, self.transformers)

    def save(self, root: str, name: str = "its_dataset"):
        """
        Save dataset.
        """
        # 1) Create main directory
        main_path = os.path.join(root, name)
        os.mkdir(main_path)
        # 2) Write config file
        config_dict = json.dumps(self.config.dictionary(), indent=4)
        with open(os.path.join(main_path, "config.json"), "w") as outfile:
            outfile.write(config_dict)
        # 3) Write ITS's
        for i, its in enumerate(self.data):
            if its.name == None or its.name == "":
                name = i
            else:
                name = its.name
            # 3.1) Create ITS folder
            its_path = os.path.join(main_path, name)
            os.mkdir(its_path)
            # 3.2) Write data.csv
            its.series.to_csv(os.path.join(its_path, self.config.data_file_name + ".csv"),
                              sep = self.config.data_separator,
                              header = self.config.data_colnames,
                              index = False)
            # 3.3) Write features.json
            fdict = {}
            for feature in its.features:
                fdict[feature.name] = feature.value
            fdict = json.dumps(fdict, indent=4)
            with open(os.path.join(its_path, self.config.features_file_name + ".json"), "w") as outfile:
                outfile.write(fdict)

    def __getitem__(self, index):
        its = self.data.__getitem__(index)
        for transformer in self.transformers:
            its = transformer.transform(its=its)
        return its
    
    def __len__(self):
        return self.size()

# -----------------
# --- FUNCTIONS ---
# -----------------

def parse_feature(name: str, value, config: ITSDatasetConfig):
    if isinstance(value, bool):
        return BooleanFeature(name=name, value=value)
    if isinstance(value, int) or isinstance(value, float):
        return NumericFeature(name=name, value=value)
    if isinstance(value, str):
        return CategoricalFeature(name=name, 
                                  value=value, 
                                  known_values=config.knownValues(name=name),
                                  one_hot=config.oneHot(name=name))
    if isinstance(value, pd.Series):
        return SeriesFeature(value=value, name=name)
    if isinstance(value, list):
        return VectorFeature(name=name, value=value)
    
    return Feature(name=name, value=value)
    # raise ParsingError("Error on feature {}: unsupported type.".format(name))

def read_its(root: str, config: ITSDatasetConfig):
    """
    Read an `InformedTimeSeries` from the given directory.
    """
    # Read time series
    data_path = os.path.join(root, config.data_file_name + ".csv")
    series = pd.read_csv(data_path, 
                         sep=config.data_separator,
                         header=config.data_header,
                         names=config.data_colnames,
                         dtype=config.data_type)
    # Data train and target
    data_train = []
    data_target = []
    for col in series.columns:
        if config.isTrain(col):
            data_train.append(col)
        if config.isTarget(col):
            data_target.append(col)
    # Read features
    features = []
    features_train = []
    features_target = []
    features_path = os.path.join(root, config.features_file_name + ".json")
    with open(features_path, "r") as features_file:
        ft_object = json.load(features_file)
        for key in ft_object.keys():
            parsed = parse_feature(name=key, value=ft_object[key], config=config)
            features.append(parsed)
            if config.isTrain(key):
                features_train.append(key)
            if config.isTarget(key):
                features_target.append(key)
    # File name
    its_name = root[root.rindex(os.sep) + 1:]

    return InformedTimeSeries(series=series,
                              features=features,
                              data_train=data_train,
                              data_target=data_target,
                              features_train=features_train,
                              features_target=features_target,
                              name = its_name)

def load_its_dataset(root: str, config: ITSDatasetConfig = None):
    # List all sub-directories in the root folder.
    subdir = os.walk(root)
    data = []
    for i, item in enumerate(subdir):
        if i == 0:
            if config == None:
                # Load datset configuration file.
                _, _, files = item
                if "config.json" in files:
                    print("Found config.json .")
                    config = ITSDatasetConfig.fromFile(file = os.path.join(root, "config.json"))
                else:
                    config = ITSDatasetConfig.default()
                    print("config.json not found. Dafault configuration will be used.")
        else:
            dir_path = item[0]
            data.append(read_its(dir_path, config))
    
    return InformedTimeSeriesDataset(data=data, config=config)
