# --------------
# --- IMPORT ---
# --------------

import random
import pandas as pd
import numpy as np
from scipy import interpolate
from machawai.ml.data import InformedTimeSeries, NumericFeature, SeriesFeature, VectorFeature

# ---------------
# --- CLASSES ---
# ---------------

class Transformer():

    def __init__(self) -> None:
        pass

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        raise NotImplementedError()
    
# >>> NORMALIZATION <<<

class MinMaxFetaureNormalizer(Transformer):

    def __init__(self, features:'list[str]', df: pd.DataFrame, inplace: bool = False) -> None:
        super().__init__()
        self.features = features
        self.df = df
        self.inplace = inplace

    def min(self, fname: str):
        return self.df.loc["min", fname]
    
    def max(self, fname: str):
        return self.df.loc["max", fname]

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        for fname in self.features:
            feat = its.getFeature(fname)
            if feat.isCategorical():
                fval = feat.encode()
                fval = (fval - self.min(fname)) / (self.max(fname) - self.min(fname))
                nfeat = NumericFeature(name=fname, value=fval)
                its.dropFeature(fname)
                its.addFeature(nfeat)
            else:
                feat.value = (feat.value - self.min(fname)) / (self.max(fname) - self.min(fname))
        return its
    
class MinMaxSeriesNormalizer(Transformer):

    def __init__(self, df: pd.DataFrame, inplace: bool = False, colnames: 'list[str]' = None) -> None:
        super().__init__()
        self.df = df
        self.inplace = inplace
        self.colnames = colnames

    def min(self, colname: str):
        return self.df.loc["min", colname]
    
    def max(self, colname: str):
        return self.df.loc["max", colname]

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        if self.colnames == None:
            cols = its.getColnames()
        else:
            cols = self.colnames
        for col in cols:
            its.series[col] = (its.series[col] - self.min(col)) / (self.max(col) - self.min(col))
        return its
    
    def revert(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        if self.colnames == None:
            cols = its.getColnames()
        else:
            cols = self.colnames
        for col in cols:
            its.series[col] = its.series[col] * (self.max(col) - self.min(col)) + self.min(col)
        return its
    
# >>> RESIZE <<<

class CutSeriesToMaxIndex(Transformer):

    def __init__(self, 
                 colname: str, 
                 include_features: 'list[str]' = [], 
                 inplace: bool = False) -> None:
        super().__init__()
        self.colname = colname
        self.include_features = include_features
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        idx_max = its.series[self.colname].argmax()
        its.series = its.series[:idx_max + 1]
        for fname in self.include_features:
            feat = its.getFeature(fname)
            if not isinstance(feat, SeriesFeature):
                raise ValueError("CutSeriesToMax: only SeriesFeature can be transformed.")
            feat.value = feat.value[:idx_max + 1]
        return its

class CutSeriesToFeaturePoint(Transformer):

    def __init__(self, 
                 use_feature: str,
                 include_cut_point: bool = True,
                 include_features: 'list[str]' = [], 
                 inplace: bool = False) -> None:
        super().__init__()
        self.use_feature = use_feature
        self.include_cut_point = include_cut_point
        self.include_features = include_features
        self.inplace = inplace

    def getCutPoint(self, its: InformedTimeSeries) -> InformedTimeSeries:
        feat = its.getFeature(self.use_feature)
        if isinstance(feat, NumericFeature):
            return feat.value
        raise TypeError("CutSeriesTail: only NumericFeature can be used to cut the series.")

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        cut_point = self.getCutPoint(its=its)
        if self.include_cut_point:
            its.series = its.series.iloc[:cut_point + 1]
        else:
            its.series = its.series.iloc[:cut_point]

        for fname in self.include_features:
            feat = its.getFeature(fname)
            if not isinstance(feat, SeriesFeature):
                raise ValueError("CutSeriesTail: only SeriesFeature can be transformed.")
            if self.include_cut_point:
                feat.value = feat.value.iloc[:cut_point + 1]
            else:
                feat.value = feat.value.iloc[:cut_point]
        return its

class CutSeriesTail(Transformer):

    def __init__(self, 
                 tail_p: float = 0.0, 
                 include_features: 'list[str]' = [], 
                 use_feature: str = None, 
                 inplace: bool = False) -> None:
        super().__init__()
        self.use_feature = use_feature
        self.tail_p = tail_p
        self.include_features = include_features
        self.inplace = inplace

    def getTailP(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if self.use_feature != None:
            feat = its.getFeature(self.use_feature)
            if isinstance(feat, NumericFeature):
                return feat.value
            raise TypeError("CutSeriesTail: only NumericFeature can be used to cut the series.")
        return self.tail_p

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        tail_p = self.getTailP(its=its)

        series_length = its.series.shape[0]
        to_cut = int(series_length * tail_p)      
        its.series = its.series.iloc[:series_length - to_cut]

        for fname in self.include_features:
            feat = its.getFeature(fname)
            if not isinstance(feat, SeriesFeature):
                raise ValueError("CutSeriesTail: only SeriesFeature can be transformed.")
            series_length = feat.value.shape[0]
            to_cut = int(series_length * tail_p)   
            feat.value = feat.value.iloc[:series_length - to_cut]
        return its

class CutSeries(Transformer):

    def __init__(self, cut_size: int = None, min_size: int = 1, max_size: int = 100, inplace: bool = False) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.min_size = min_size
        self.max_size = max_size
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        series_length = its.series.shape[0]
        if self.cut_size == None:
            cut_size = random.randint(self.min_size, self.max_size)
        else:
            cut_size = self.cut_size
        start_point = random.randint(0, series_length - cut_size)
        its.series = its.series.loc[start_point: start_point + cut_size - 1]
        return its

class CutSeriesWithPadding(Transformer):

    def __init__(self, cut_size: int = None, min_size: int = 1, max_size: int = 100, pad_value: 'float | str' = 0.0, max_padding: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.min_size = min_size
        self.max_size = max_size
        self.pad_value = pad_value
        self.max_padding = max(0, min(1, max_padding))
        self.inplace = inplace

    def getPaddingBefore(self, start_point: int, its: InformedTimeSeries) -> pd.DataFrame:
        if isinstance(self.pad_value, int) or isinstance(self.pad_value, float):
            padding = np.full(shape = (abs(start_point), its.series.shape[1]), fill_value=self.pad_value)
            padding_index = np.arange(start_point, 0)
            return pd.DataFrame(padding, columns=its.getColnames(), index=padding_index)    
        elif isinstance(self.pad_value, str):
            if self.pad_value == "repeat":
                padding = its.series.iloc[0].values
                padding = np.expand_dims(padding, 0)
                padding = np.repeat(padding, abs(start_point), axis = 0)
                padding_index = np.arange(start_point, 0)
                return pd.DataFrame(padding, columns=its.getColnames(), index=padding_index) 
        raise ValueError("Invalid padding value.")
    
    def getPaddingAfter(self, start_point: int, cut_size: int, series_length: int, its: InformedTimeSeries) -> pd.DataFrame:
        if isinstance(self.pad_value, int) or isinstance(self.pad_value, float):
            padding = np.full(shape = (cut_size - (series_length - start_point), its.series.shape[1]), fill_value=self.pad_value)
            padding_index = np.arange(series_length, start_point + cut_size)
            return pd.DataFrame(padding, columns=its.getColnames(), index=padding_index)    
        elif isinstance(self.pad_value, str):
            if self.pad_value == "repeat":
                padding = its.series.iloc[-1].values
                padding = np.expand_dims(padding, 0)
                padding = np.repeat(padding, cut_size - (series_length - start_point), axis = 0)
                padding_index = np.arange(series_length, start_point + cut_size)
                return pd.DataFrame(padding, columns=its.getColnames(), index=padding_index) 
        raise ValueError("Invalid padding value.")

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        series_length = its.series.shape[0]
        if self.cut_size == None:
            cut_size = random.randint(self.min_size, self.max_size)
        else:
            cut_size = self.cut_size

        max_padding_size = int(cut_size * self.max_padding)
        min_start_point = 1 if max_padding_size == 0 else 1 - max_padding_size
        max_start_point = series_length - 2 if cut_size - max_padding_size < 2 else series_length - (cut_size - max_padding_size) 
        start_point = random.randint(min_start_point , max_start_point)

        #start_point = random.randint(1 - cut_size, series_length - 2)
        if start_point < 0:
            # Add padding before
            padding = self.getPaddingBefore(start_point=start_point, its=its)
            its.series = pd.concat([padding, its.series])
        elif start_point > series_length - cut_size:
            # Add padding after
            padding = self.getPaddingAfter(start_point=start_point, cut_size=cut_size, series_length=series_length, its=its)
            its.series = pd.concat([its.series, padding])
        its.series = its.series.loc[start_point: start_point + cut_size - 1]
        return its

class InterpolateSeries(Transformer):

    def __init__(self, 
                 x_label: str, 
                 y_label: str,
                 new_x_label: str = None, 
                 size: int = None, 
                 interp_x_name: str = "INTERP_X",
                 interp_y_name: str = "INTERP_Y",
                 inplace: bool = False) -> None:
        super().__init__()
        self.x_label = x_label
        self.y_label = y_label
        self.new_x_label = new_x_label
        self.size = size
        self.inplace = inplace
        self.interp_x_name = interp_x_name
        self.interp_y_name = interp_y_name

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        x = its.getColumn(self.x_label)
        x = x.dropna(axis = 0)
        y = its.getColumn(self.y_label)
        y = y.dropna(axis = 0)
        if self.new_x_label != None:
            new_x = its.getColumn(self.new_x_label)
            new_x = new_x.dropna(axis=0)
        else:
            if self.size == None:
                raise ValueError("Invalid size value.")
            new_x = np.linspace(x.min(), x.max(), self.size)
        new_y = np.interp(new_x, x, y)
        
        series_length = its.series.shape[0]
        if new_y.shape[0] < series_length:
            d = series_length - new_y.shape[0]
            new_y = np.concatenate([new_y, np.full(shape = (d,), fill_value = np.nan)])
            new_x = np.concatenate([new_x, np.full(shape = (d,), fill_value = np.nan)])
        elif new_y.shape[0] > series_length:
            d = new_y.shape[0] - series_length
            padding = np.full(shape = (d, its.series.shape[1]), fill_value = np.nan)
            padding_index = np.arange(series_length, series_length + d)
            padding = pd.DataFrame(padding, columns=its.getColnames(), index=padding_index)  
            its.series = pd.concat([its.series, padding])
        
        its.series[self.interp_y_name] = new_y
        if self.new_x_label == None:
            its.series[self.interp_x_name] = new_x

        return its

class BSplineInterpolate(Transformer):

    def __init__(self,
                 size: int,
                 s: float = 0.0,
                 k: int = 1,
                 save_old_size_as: str = None,
                 inplace: bool = False) -> None:
        super().__init__()
        self.size = size
        self.s = s
        self.k = k
        self.save_old_size_as = save_old_size_as
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        if self.save_old_size_as != None and \
           self.save_old_size_as != "" and \
           not its.hasFeature(self.save_old_size_as):
            its.addFeature(NumericFeature(name = self.save_old_size_as, value=its.series.shape[0]))
        curve = its.series.values
        tck, u = interpolate.splprep(curve.T, s = self.s, k = self.k)
        u_new = np.linspace(u.min(), u.max(), self.size)
        interp_curve = interpolate.splev(u_new, tck, der=0)
        new_series = pd.DataFrame(interp_curve).transpose()
        new_series.columns = its.series.columns
        its.series = new_series

        return its

class CustomInterpolation(Transformer):

    def __init__(self,
                 size: int,
                 disp_label: str,
                 load_label: str,
                 strain_label: str = None,
                 kind: str = 'cubic',
                 save_old_values: 'list[str]' = [],
                 inplace: bool = False) -> None:
        super().__init__()
        self.size = size
        self.disp_label = disp_label
        self.load_label = load_label
        self.strain_label = strain_label
        self.inplace = inplace
        self.kind = kind
        self.save_old_values = save_old_values

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        t = np.arange(0, its.series.shape[0], 1)
        disp = its.getColumn(self.disp_label).values
        load = its.getColumn(self.load_label).values
        if self.strain_label != None:
            strain = its.getColumn(self.strain_label).values

        y = np.vstack([disp[None, ...], load[None, ...]])
        spline = interpolate.interp1d(t, y, kind=self.kind)
        new_t = np.linspace(t.min(), t.max(), self.size)
        new_y = spline(new_t)
        new_disp = new_y[0,:]
        new_load = new_y[1,:]

        if self.strain_label != None:
            y_2 = np.vstack([strain[None, ...], load[None, ...]])
            spline_2 = interpolate.interp1d(t, y_2, kind=self.kind)
            new_y_2 = spline_2(new_t)
            new_strain = new_y_2[0,:]
            new_load_2 = new_y_2[1,:]

            assert (new_load == new_load_2).sum() == self.size

        if self.strain_label != None:
            new_series = pd.DataFrame({
                self.disp_label : new_disp,
                self.load_label : new_load,
                self.strain_label : new_strain
            })
        else: 
            new_series = pd.DataFrame({
                self.disp_label : new_disp,
                self.load_label : new_load,
            })
        for label in self.save_old_values:
            name = "OLD_"+label
            value = its.getColumn(label).tolist()
            its.addFeature(VectorFeature(name = name, value = value))
        its.series = new_series

        return its

# >>> GENERIC <<<

class RenameTraining(Transformer):

    def __init__(self, old: str, new: str, inplace: bool = False) -> None:
        super().__init__()
        self.old = old
        self.new = new
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        its.untrain(self.old)
        its.setTraining(self.new)
        return its
    
class RenameTarget(Transformer):

    def __init__(self, old: str, new: str, inplace: bool = False) -> None:
        super().__init__()
        self.old = old
        self.new = new
        self.inplace = inplace

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        if not self.inplace:
            its = its.copy()
        its.untarget(self.old)
        its.setTarget(self.new)
        return its