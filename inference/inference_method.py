import gc
from typing import Literal, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.data_utils import TabularInferenceDataset
from utils.loading import load_model

from utils.retrieval_utils import RelabelRetrievalY


class InferenceResultWithRetrieval:
    def __init__(self,
                 model: torch.nn.Module,
                 sample_selection_type: Literal["AM", "DDP"] = "AM",
                 ):
        self.model = model
        self.sample_selection_type = sample_selection_type
        self.dataset = None

    def _prepare_data(self,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      attention_score: np.ndarray = None,
                      retrieval_len: int = 2000
                      ) -> TabularInferenceDataset:
        if self.sample_selection_type == "AM":
            use_retrieval = True
        else:
            use_retrieval = False
        dataset = TabularInferenceDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            attention_score=attention_score,
            retrieval_len=retrieval_len,
            use_retrieval=use_retrieval
        )
        return dataset

    def inference(self,
                  X_train: torch.Tensor = None,
                  y_train: torch.Tensor = None,
                  X_test: torch.Tensor = None,
                  dataset: TabularInferenceDataset = None,
                  attention_score: np.ndarray | torch.Tensor = None,
                  retrieval_len: int = 2000,
                  dynamic_ratio: float = None,
                  task_type: Literal["reg", "cls"] = "reg"):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        if isinstance(retrieval_len, str):
            if retrieval_len == "dynamic":
                if dynamic_ratio is not None:
                    retrieval_len = int(dynamic_ratio * X_train.shape[0] / len(torch.unique(y_train)))
                else:
                    retrieval_len = int(X_train.shape[0] / len(torch.unique(y_train)))
        
        if isinstance(retrieval_len, float):
            self.retrieval_len = int(retrieval_len * X_train.shape[0])
        else:
            self.retrieval_len = retrieval_len

        if dataset is None:
            dataset = self._prepare_data(X_train, y_train, X_test, attention_score, self.retrieval_len)

        outputs = []
        dataloader = DataLoader(dataset,
                                batch_size=16,
                                shuffle=False,
                                drop_last=False
                                )
        for data in dataloader:
            with (
                torch.autocast(device.type, enabled=True),
                torch.inference_mode(),
            ):
                if self.sample_selection_type == "DDP":
                    # This branch is not supported for single-card inference, as it assumes DDP
                    # and will likely not work as intended.
                    # Consider removing or modifying this logic if not needed.
                    X_test_item = data["X_test"].unsqueeze(0)
                    X_train_item = data["X_train"].unsqueeze(0)
                    Y_train_item = data["y_train"].unsqueeze(0).unsqueeze(-1)
                    x_ = torch.cat([X_train_item, X_test_item], dim=1).to(device)
                    y_ = Y_train_item.squeeze(-1).to(device)
                    
                    output = self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)
                else:
                    X_train_item = data["X_train"].unsqueeze(0).to(device)
                    X_test_item = data["X_test"].unsqueeze(0).to(device)
                    y_ = data["y_train"].unsqueeze(0).to(device)
                    x_ = torch.cat([X_train_item, X_test_item], dim=1)
                    
                    if task_type == "cls":
                        relabel = RelabelRetrievalY(y_.squeeze(0).cpu())
                        y_ = relabel.transform_y().unsqueeze(0).to(device)

                    output = self.model(x=x_, y=y_.squeeze(-1), eval_pos=y_.shape[1], task_type=task_type)
                    
                    if len(output.shape) == 3:
                        output = output.view(-1, output.shape[-1])
                    
                    if task_type == "cls":
                        output = output.cpu().numpy()
                        output = relabel.inverse_transform_y(output)
                        output = torch.tensor(output, dtype=torch.float32, device=device)

            outputs.append(output.cpu())
            del output
            gc.collect()
            torch.cuda.empty_cache()

        outputs = torch.cat(outputs, dim=0)
        return outputs.squeeze(0)


class InferenceAttentionMap:
    def __init__(self,
                 model_path: str,
                 calculate_feature_attention: bool = False,
                 calculate_sample_attention: bool = False,
                 ):
        self.calculate_feature_attention = calculate_feature_attention
        self.calculate_sample_attention = calculate_sample_attention
        self.model = load_model(model_path, calculate_feature_attention=calculate_feature_attention,
                           calculate_sample_attention=calculate_sample_attention)
        self.dataset = None

    def _prepare_data(self,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      ) -> TabularInferenceDataset:
        dataset = TabularInferenceDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            use_retrieval=False
        )
        return dataset

    def inference(self,
                  X_train: torch.Tensor | np.ndarray,
                  y_train: torch.Tensor | np.ndarray,
                  X_test: torch.Tensor | np.ndarray,
                  task_type: Literal["reg", "cls"] = "reg") -> tuple[torch.Tensor | None, torch.Tensor | None]:
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        
        dataset = self._prepare_data(X_train.cpu(), y_train.cpu(), X_test.cpu())
        dataloader = DataLoader(dataset,
                                batch_size=16,
                                shuffle=False,
                                drop_last=False
                                )

        X_train_full = X_train.to(device)
        y_train_full = y_train.to(device)

        feature_attentions = []
        sample_attentions = []
        
        with torch.autocast(device.type, enabled=True), torch.inference_mode():
            for data in dataloader:
                X_test_batch = data["X_test"].to(device)

                x_ = torch.cat([X_train_full, X_test_batch], dim=0).unsqueeze(0)
                y_ = y_train_full.unsqueeze(0)

                output, feature_attention, sample_attention = self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)

                if self.calculate_sample_attention:
                    # Squeeze and append the sample attention scores for the current batch
                    # The sample attention tensor is likely in the shape of [1, 1, 1, num_test_samples]
                    # We only care about the test samples, which is the latter part of the sequence.
                    # We get the test samples, squeeze them and append them to the list.
                    sample_attentions.append(sample_attention.squeeze(0)[:, y_train_full.shape[0]:].squeeze())
                    
                if self.calculate_feature_attention:
                    # Squeeze and append the feature attention scores
                    feature_attentions.append(feature_attention.squeeze(0)[y_train_full.shape[0]:])
                
                del output, sample_attention, feature_attention
                gc.collect()
                torch.cuda.empty_cache()

        final_feature_attention = torch.cat(feature_attentions, dim=0) if feature_attentions else None
        final_sample_attention = torch.cat(sample_attentions, dim=0) if sample_attentions else None

        return final_feature_attention, final_sample_attention