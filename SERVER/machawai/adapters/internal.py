"""
    Classes and functions for internal packages communication.
"""

# --------------
# --- IMPORT ---
# --------------

from machawai.tensile import TensileTest
from machawai.ml.data import InformedTimeSeries, parse_feature, ITSDatasetConfig, InformedTimeSeriesDataset

# -----------------
# --- FUNCTIONS ---
# -----------------

def parse_tensile_test(ttest: TensileTest, config: ITSDatasetConfig) -> InformedTimeSeries:
    series = ttest.getFullData()
    features = []
    labels = ttest.labels()
    for label in labels:
        if not label in series.columns:
            feature = parse_feature(name=label,
                                    value=ttest.get_by_label(label), 
                                    config=config)
            features.append(feature)

    return InformedTimeSeries(series=series,
                              features=features,
                              data_train=config.data_train,
                              data_target=config.data_target,
                              features_train=config.features_train,
                              features_target=config.features_target,
                              name=ttest.filename)

def parse_tensile_test_collection(collection: 'list[TensileTest]', config: ITSDatasetConfig) -> InformedTimeSeriesDataset:
    data = []
    for ttest in collection:
        its = parse_tensile_test(ttest=ttest, config=config)
        data.append(its)
    dataset = InformedTimeSeriesDataset(data = data, config = config)
    return dataset