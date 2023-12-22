"""
    Classes and functions for pytorch integration.
"""
# --------------
# --- IMPORT ---
# --------------

from machawai.ml.data import InformedTimeSeries, InformedTimeSeriesDataset
from machawai.ml.transformer import Transformer, CutSeries, CutSeriesWithPadding
import torch
import numpy as np
import random
from tqdm import tqdm

# ---------------
# --- CLASSES ---
# ---------------

class ITSWrapper(InformedTimeSeries):

    def __init__(self, its: InformedTimeSeries, add_batch_dim: bool = True, device: str = "cpu", dtype = torch.float32) -> None:
        super().__init__(series=its.series, 
                         features=its.features, 
                         data_train=its.data_train, 
                         data_target=its.data_target, 
                         features_train=its.features_train, 
                         features_target=its.features_target, 
                         name=its.name)
        self.its = its
        self.device = device
        self.dtype = dtype
        self.add_batch_dim = add_batch_dim

    def copy(self) -> 'ITSWrapper':
        return ITSWrapper(its=self.its.copy(),
                          add_batch_dim=self.add_batch_dim,
                          device=self.device,
                          dtype=self.dtype)

    def feature_tensor(self, name: str, add_batch_dim: bool = False):
        """
            Retrun the encoded feature as a tensor given its name.
        """
        feature = self.getFeature(name = name).encode()
        tensor = torch.tensor(feature, device=self.device, dtype=self.dtype)
        if add_batch_dim:
            tensor = tensor[None, ...]
        return tensor

    def train_series_tensor(self, add_batch_dim: bool = False):
        """
            Return the train series as a tensor.
        """
        train_series = self.getTrainSeries().values
        tensor = torch.tensor(train_series, device=self.device, dtype=self.dtype)
        if add_batch_dim:
            tensor = tensor[None, ...]
        return tensor
    
    def train_features_tensor(self, add_batch_dim: bool = False):
        features = np.array([])
        for i, feat in enumerate(self.its.getTrainFeatures()):
            if i == 0:
                features = feat.encode()
            else:
                features = np.hstack([features, feat.encode()])
        if features.any():
            features = torch.tensor(features, device=self.device, dtype=self.dtype)
            if add_batch_dim:
                features = features[None, ...]
        else:
            features = torch.tensor([], device=self.device)
        return features
    
    def target_series_tensor(self, add_batch_dim: bool = False):
        s_tensor = self.getTargetSeries().values
        s_tensor = torch.tensor(s_tensor, device=self.device, dtype=self.dtype)
        if add_batch_dim:
            s_tensor = s_tensor[None, ...]
        return s_tensor
    
    def target_features_tensor(self, add_batch_dim: bool = False):
        features = np.array([])
        for i, feat in enumerate(self.its.getTargetFeatures()):
            if i == 0:
                features = feat.encode()
            else:
                features = np.hstack([features, feat.encode()])
        if features.any():
            features = torch.tensor(features, device=self.device, dtype=self.dtype)
            if add_batch_dim:
                features = features[None, ...]
        else:
            features = torch.tensor([], device=self.device)
        return features

    def X(self):
        train_series = self.train_series_tensor(add_batch_dim=self.add_batch_dim)
        train_features = self.train_features_tensor(add_batch_dim=self.add_batch_dim)
        if train_series.any():
            if train_features.any():
                return train_series, train_features
            return train_series
        if train_features.any():
            return train_features
        return torch.tensor([], device=self.device)
    
    def Y(self):
        target_series = self.target_series_tensor(self.add_batch_dim)
        target_features = self.target_features_tensor(self.add_batch_dim)
        if target_series.any():
            if target_features.any():
                return target_series, target_features
            return target_series
        if target_features.any():
            return target_features
        return torch.tensor([], device=self.device)
    
    def __iter__(self):
        return iter((self.X(), self.Y()))

class ITSBatch():

    def __init__(self, batch: 'list[ITSWrapper]') -> None:
        self.batch = batch
        self.size = len(self.batch)

    def copy(self) -> 'ITSBatch':
        return ITSBatch(batch=[b.copy() for b in self.batch])
    
    def feature_tensor(self, name: str):
        tensor = self.batch[0].feature_tensor(name = name, add_batch_dim = True)
        for i in range(1, self.size):
            tensor = torch.cat([tensor, self.batch[i].feature_tensor(name = name, add_batch_dim = True)], dim=0)
        return tensor

    def train_series_tensor(self):
        tensor = self.batch[0].train_series_tensor(add_batch_dim=True)
        if not tensor.any():
            return torch.tensor([], device = tensor.device)
        for i in range(1, self.size):
            tensor = torch.cat([tensor, self.batch[i].train_series_tensor(True)], dim=0)
        return tensor
    
    def train_features_tensor(self):
        tensor = self.batch[0].train_features_tensor(add_batch_dim=True)
        if not tensor.any():
            return torch.tensor([], device = tensor.device)
        for i in range(1, self.size):
            tensor = torch.cat([tensor, self.batch[i].train_features_tensor(True)], dim=0)
        return tensor
    
    def target_series_tensor(self):
        tensor = self.batch[0].target_series_tensor(add_batch_dim=True)
        if not tensor.any():
            return torch.tensor([], device = tensor.device)
        for i in range(1, self.size):
            tensor = torch.cat([tensor, self.batch[i].target_series_tensor(True)], dim=0)
        return tensor
    
    def target_features_tensor(self):
        tensor = self.batch[0].target_features_tensor(add_batch_dim=True)
        if not tensor.any():
            return torch.tensor([], device = tensor.device)
        for i in range(1, self.size):
            tensor = torch.cat([tensor, self.batch[i].target_features_tensor(True)], dim=0)
        return tensor

    def X(self):
        train_series = self.train_series_tensor()
        train_features = self.train_features_tensor()
        if train_series.any():
            if train_features.any():
                return train_series, train_features
            return train_series
        if train_features.any():
            return train_features
        return torch.tensor([], device=train_series.device)
    
    def Y(self):
        target_series = self.target_series_tensor()
        target_features = self.target_features_tensor()
        if target_series.any():
            if target_features.any():
                return target_series, target_features
            return target_series
        if target_features.any():
            return target_features
        return torch.tensor([], device=target_series.device)

    def __iter__(self):
        return iter((self.X(), self.Y()))

class ITSBatchDataGenerator():

    def __init__(self, dataset: InformedTimeSeriesDataset, batch_size: int = 8, device: str = "cpu", dtype = torch.float32) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.transformers = []
        self.batch_transformers = []
        self.device = device
        self.dtype = dtype

    def size(self) -> int:
        return self.dataset.size() // self.batch_size
    
    def shuffle(self):
        self.dataset.shuffle()
    
    def __getitem__(self, index):
        batch = []
        b_start = index * self.batch_size
        b_end = (index + 1) * self.batch_size
        for i in range(b_start, b_end):
            its = self.dataset[i]
            for transformer in self.transformers:
                its = transformer.transform(its=its)
            its = ITSWrapper(its=its, add_batch_dim=True,
                             device=self.device, dtype=self.dtype)
            batch.append(its)
        batch = ITSBatch(batch=batch)
        for transformer in self.batch_transformers:
            batch = transformer.transform(its_batch=batch)
        return batch
    
    def __len__(self):
        return self.size()

class BatchLoss():

    def __init__(self) -> None:
        pass

    def __call__(self, Y: torch.Tensor, Yp: torch.Tensor, X: torch.Tensor = None, batch: ITSBatch = None) -> torch.Tensor:
        raise NotImplementedError()
    
class BatchMetric():

    def __init__(self, name: str = None) -> None:
        self.name = name

    def __call__(self, Y: torch.Tensor, Yp: torch.Tensor, X: torch.Tensor = None, batch: ITSBatch = None) -> float:
        raise NotImplementedError()

# TODO: rivedere gesione Transformers

class WrapTransformer(Transformer):

    def __init__(self, add_batch_dim: bool = True, device: str = "cpu", dtype = torch.float32) -> None:
        super().__init__()
        self.add_batch_dim = add_batch_dim
        self.device = device
        self.dtype = dtype

    def transform(self, its: InformedTimeSeries) -> InformedTimeSeries:
        return ITSWrapper(its=its, 
                          add_batch_dim=self.add_batch_dim,
                          device=self.device,
                          dtype=self.dtype)
    
class BatchTransformer():

    def __init__(self) -> None:
        pass

    def transform(self, batch: ITSBatch):
        raise NotImplementedError()
    
class BatchCutSeries(BatchTransformer):

    def __init__(self, min_size: int, max_size: int = None, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.min_size = min_size
        self.max_size = max_size

    def transform(self, its_batch: ITSBatch):
        if not self.inplace:
            its_batch = its_batch.copy()
        if self.max_size == None:
            max_size = np.min([its.series.shape[0] for its in its_batch.batch])
        else:
            max_size = self.max_size
        cut_size = random.randint(self.min_size, max_size)
        cut_series = CutSeries(cut_size=cut_size, inplace=True)
        for i in range(its_batch.size):
            cut_series.transform(its_batch.batch[i])
        return its_batch
    
class BatchCutSeriesWithPadding(BatchTransformer):

    def __init__(self, min_size: int, max_size: int = None, pad_value: 'float | str' = 0.0, max_padding: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.pad_value = pad_value
        self.max_padding = max_padding
        self.min_size = min_size
        self.max_size = max_size

    def transform(self, its_batch: ITSBatch):
        if not self.inplace:
            its_batch = its_batch.copy()
        if self.max_size == None:
            max_size = np.min([its.series.shape[0] for its in its_batch.batch])
        else:
            max_size = self.max_size
        cut_size = random.randint(self.min_size, max_size)
        cut_series = CutSeriesWithPadding(cut_size=cut_size, pad_value=self.pad_value, max_padding=self.max_padding, inplace=True)
        for i in range(its_batch.size):
            cut_series.transform(its_batch.batch[i])
        return its_batch

# ----------------
# --- FUNCTION ---
# ----------------

def batch_train(model: torch.nn.Module, 
                generator: ITSBatchDataGenerator, 
                epochs: int, 
                batch_loss: BatchLoss,
                optimizer: torch.optim.Optimizer = None,
                lr_scheduler = None,
                shuffle: bool = True,
                verbose: int = 1,
                metrics: 'list[BatchMetric]' = []):
    # INIT PHASE
    num_batches = generator.size()
    # Init. history dictionary
    history = {"loss": [],
               "valid_loss": []}
    for metric in metrics:
        history[metric.name] = []
        history["valid_"+metric.name] = []
    epochs = tqdm(range(epochs), desc="Progress") if verbose == 0 else range(epochs)
    model.train()
    # START TRAINING
    for epoch in epochs:
        # START EPOCH
        if shuffle:
            generator.shuffle()
        data_generator = tqdm(generator, desc="Epoch {}".format(epoch + 1)) if verbose > 0 else generator
        for i, batch in enumerate(data_generator):
            if num_batches > 1 and i == num_batches - 1:
                train_phase = False
            else:
                train_phase = True
            if train_phase:
                # Train phase
                X, Y = batch
                # Predict
                Yp = model(X)
                # Compute loss
                loss = batch_loss(Y=Y, Yp=Yp, X=X, batch=batch)
                # Compute metrics
                metric_buffer = []
                for i, metric in enumerate(metrics):
                    metric_value = metric(Y=Y, Yp=Yp, X=X, batch=batch)
                    metric_buffer.append(metric_value)
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Schedule learning rate if needed
                if lr_scheduler != None:
                    lr_scheduler.step()
                # Show progress
                if verbose > 0:
                    postfix = {"LOSS": loss.item()}
                    for i, metric in enumerate(metrics):
                        postfix[metric.name] = metric_buffer[i]
                    data_generator.set_postfix(postfix)
            else:
                # Validation phase
                model.eval()
                X, Y = batch
                # Predict
                Yp = model(X)
                # Compute loss
                valid_loss = batch_loss(Y=Y, Yp=Yp, X=X, batch=batch)
                # Compute metrics
                valid_metric_buffer = []
                for i, metric in enumerate(metrics):
                    metric_value = metric(Y=Y, Yp=Yp, X=X, batch=batch)
                    valid_metric_buffer.append(metric_value)
                # Show progress
                if verbose > 0:
                    postfix["VALID. LOSS"] = valid_loss.item()
                    for i, metric in enumerate(metrics):
                        postfix["VALID. " + metric.name] = valid_metric_buffer[i] 
                    data_generator.set_postfix(postfix)
                model.train()
        # END EPOCH
        # Update history
        history["loss"].append(loss.item())
        history["valid_loss"].append(valid_loss.item())
        for i, metric in enumerate(metrics):
            history[metric.name].append(metric_buffer[i])
            history["valid_"+metric.name].append(valid_metric_buffer[i]) 
        # Show Progress
        if verbose == 0:
            postfix = {"LOSS": loss.item()}
            for i, metric in enumerate(metrics):
                postfix[metric.name] = metric_buffer[i]
            postfix["VALID. LOSS"] = valid_loss.item()
            for i, metric in enumerate(metrics):
                postfix["VALID. " + metric.name] = valid_metric_buffer[i]
            epochs.set_postfix(postfix)
    # END TRAINING
    return history
