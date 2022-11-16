from typing import List, Tuple, Dict
import torch


class DataCollatorTwoInputsPacked:
    def __init__(self, padding_idx: int):
        self.padding_idx = padding_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        x: ([T, M], [T, M])
        """
        max_length = max(x[0].shape[0] for x in batch)
        N = len(batch)
        M = batch[0][0].shape[1]
        track_ids = torch.full((N, max_length, M), fill_value=self.padding_idx).long()
        artist_ids = torch.full((N, max_length, M), fill_value=self.padding_idx).long()
        for i, (t, a) in enumerate(batch):
            track_ids[i, :t.shape[0], :] = t
            artist_ids[i, :a.shape[0], :] = a
        return {
            "track_ids": track_ids,
            "artist_ids": artist_ids
        }


class DataCollatorTwoInputsUnpacked:
    def __init__(self, padding_idx: int):
        self.padding_idx = padding_idx

    def __call__(self, batch: List[Tuple[List[int], List[int], List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        """
        x: ([T, M], [T, M])
        """
        max_length = max(len(x[0]) for x in batch)
        N = len(batch)
        track_ids_x = torch.full((N, max_length), fill_value=self.padding_idx).long()
        track_ids_y = torch.full((N, max_length), fill_value=self.padding_idx).long()
        artist_ids_x = torch.full((N, max_length), fill_value=self.padding_idx).long()
        artist_ids_y = torch.full((N, max_length), fill_value=self.padding_idx).long()
        for i, (tx, ty, ax, ay) in enumerate(batch):
            track_ids_x[i, :len(tx)] = torch.tensor(tx).long()
            track_ids_y[i, :len(ty)] = torch.tensor(ty).long()
            artist_ids_x[i, :len(tx)] = torch.tensor(ax).long()
            artist_ids_y[i, :len(ty)] = torch.tensor(ay).long()
        return {
            "track_ids_x": track_ids_x,
            "track_ids_y": track_ids_y,
            "artist_ids_x": artist_ids_x,
            "artist_ids_y": artist_ids_y,
        }


class DataCollatorSingleInput:
    def __init__(self, padding_idx: int):
        self.padding_idx = padding_idx

    def __call__(self, batch: List[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        """
        x: [T]
        """
        max_length = max(len(x[0]) for x in batch)
        inputs = torch.full((len(batch), max_length), fill_value=self.padding_idx).long()
        targets = torch.full((len(batch), max_length), fill_value=self.padding_idx).long()
        for i, (x, y) in enumerate(batch):
            assert len(x) == len(y)
            inputs[i, :len(x)] = torch.tensor(x).long()
            targets[i, :len(y)] = torch.tensor(y).long()
        return {
            "inputs": inputs,
            "targets": targets
        }
