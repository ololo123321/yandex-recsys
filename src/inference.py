import os
import json
import math
import logging
from typing import List, Dict
import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod

import hydra
from omegaconf import OmegaConf

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import DataLoader

from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import CountVectorizer

from src.model import get_padding_mask
from src.datasets import DatasetV2, DatasetV3
from src.collators import DataCollatorSingleInput
from src.utils import get_track_id_to_artist_id


logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)


def load_model(model_dir, checkpoint_dir, device):
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path) as f:
        config = OmegaConf.load(f)
    model = hydra.utils.instantiate(config.model)
    checkpoint_path = os.path.join(model_dir, checkpoint_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


class BaseInferenceWrapper(ABC):
    def __init__(self, batch_size: int = 32, k: int = 100, test: bool = False):
        self.batch_size = batch_size
        self.k = k
        self.test = test

    @abstractmethod
    def __call__(self, test_data: List[List[str]], **kwargs) -> List[List[str]]: ...


class InferenceWrapperV3(BaseInferenceWrapper):
    """
    p(track) = p(track|artist,history) * p(artist|history)
    """
    def __init__(
            self,
            model_dir,
            checkpoint_dir,
            track_to_artist_path: str,
            artist_vocab_path: str,
            track_counts_path: str,
            artist_counts_path: str,
            bos_id: int = 1,
            unk_id: int = 2,
            device: str = "cuda",
            batch_size: int = 32,
            k: int = 100,
            test: bool = False
    ):
        super().__init__(batch_size=batch_size, k=k, test=test)
        self.model = load_model(model_dir=model_dir, checkpoint_dir=checkpoint_dir, device=device)

        self.collator = DataCollatorSingleInput(padding_idx=self.model.padding_idx)
        self.device = device
        self.bos_id = bos_id
        self.unk_id = unk_id

        self.track2artist = {}
        self.artist2tracks = defaultdict(list)
        with open(track_to_artist_path) as f:
            next(f)
            for line in f:
                track, artist = line.strip().split(",")
                self.track2artist[track] = artist
                self.artist2tracks[artist].append(track)

        with open(artist_vocab_path) as f:
            self.artist2id = json.load(f)

        self.id2artist = {v: k for k, v in self.artist2id.items()}

        with open(artist_counts_path) as f:
            self.artist_counts = json.load(f)

        with open(track_counts_path) as f:
            self.track_counts = json.load(f)

    def __call__(self, test_data: List[List[str]], **kwargs) -> List[List[str]]:
        batch_size = kwargs.pop("batch_size", self.batch_size)
        k = kwargs.pop("k", self.k)
        test = kwargs.pop("test", self.test)

        ds = DatasetV3(
            data=test_data,
            track2artist=self.track2artist,
            artist2id=self.artist2id,
            bos_id=self.bos_id,
            unk_id=self.unk_id,
            test=test
        )
        loader = DataLoader(
            ds,
            collate_fn=self.collator,
            batch_size=batch_size,
            shuffle=False
        )
        preds = []
        pbar = tqdm.tqdm(total=len(ds))
        idx = 0
        with torch.no_grad():
            for batch in loader:
                artist_ids = batch["inputs"].to(self.device)
                batch_size = artist_ids.shape[0]
                outputs = self.model(track_ids=None, artist_ids=artist_ids)  # [N, T, D]
                padding_mask = get_padding_mask(artist_ids, padding_idx=self.model.padding_idx)  # [N, T]
                xs = torch.arange(batch_size).to(self.device)
                ys = padding_mask.sum(1) - 1
                last_item = outputs[xs, ys, :]  # [N, D]
                logits = torch.matmul(last_item, self.model.artist_emb.weight.T)  # [N, V]
                probs_batch = torch.softmax(logits, dim=-1)  # [N, V]
                x = torch.topk(probs_batch, k, dim=-1)  # ([N, K], [N, K])
                top_probs = x.values.to("cpu").numpy()
                top_artists = x.indices.to("cpu").numpy()
                for j in range(batch_size):
                    top_tracks = self._get_top_tracks(
                        top_artists=top_artists[j],
                        top_probs=top_probs[j],
                        tracks_history=test_data[idx],
                        k=k
                    )
                    preds.append(top_tracks)
                    idx += 1
                pbar.update(batch_size)
        pbar.close()
        return preds

    def _get_top_tracks(
            self,
            top_artists: np.ndarray,
            top_probs: np.ndarray,
            tracks_history: List[str],
            k: int = 100
    ) -> List[str]:
        """
        score = p(artist|history) * p(track|artist,history)
        """
        history_set = set(tracks_history)
        res = []
        for i in range(k):
            artist_id = top_artists[i]
            p_artist = top_probs[i]
            artist = self.id2artist[artist_id]
            artist_counts_i = self.artist_counts[artist]
            new_tracks = []
            for track in self.artist2tracks[artist]:
                if track in history_set:
                    artist_counts_i -= self.track_counts[track]
                else:
                    new_tracks.append(track)
            for track in new_tracks:
                p_track = self.track_counts[track] / artist_counts_i
                score = p_artist * p_track
                res.append((track, score))
        top_tracks = [track for track, score in sorted(res, key=lambda x: -x[1])[:k]]
        return top_tracks


class InferenceWrapperJoint(BaseInferenceWrapper):
    """
    p(track) + p(artist)
    """
    def __init__(
            self,
            artist_model_dir,
            artist_checkpoint_dir,
            joint_model_dir,
            joint_checkpoint_dir,
            track_to_artist_path: str,
            track_vocab_path: str,
            artist_vocab_path: str,
            artist_vocab_path_joint: str,
            w_joint: float,
            w_artist: float = None,
            bos_id: int = 1,  # warning: bos и unk должны быть одинаковыми у двух моделей
            unk_id: int = 2,
            device: str = "cuda",
            batch_size: int = 32,
            k: int = 100,
            test: bool = False
    ):
        super().__init__(batch_size=batch_size, k=k, test=test)
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.device = device
        self.w_joint = w_joint
        if w_artist is not None:
            self.w_artist = w_artist
        else:
            assert w_joint <= 1.0
            self.w_artist = 1.0 - w_joint

        self.track2artist = {}
        with open(track_to_artist_path) as f:
            next(f)
            for line in f:
                track, artist = line.strip().split(",")
                self.track2artist[track] = artist

        with open(track_vocab_path) as f:
            self.track2id = json.load(f)
        self.id2track = {v: k for k, v in self.track2id.items()}

        with open(artist_vocab_path) as f:
            self.artist2id = json.load(f)

        with open(artist_vocab_path_joint) as f:
            self.artist2id_joint = json.load(f)

        self.model_artist = load_model(artist_model_dir, artist_checkpoint_dir, device=device)
        self.model_joint = load_model(joint_model_dir, joint_checkpoint_dir, device=device)

        self.track_id_to_artist_id = get_track_id_to_artist_id(
            track2id=self.track2id,
            artist2id=self.artist2id,
            track2artist=self.track2artist,
            num_special_tokens=self.model_joint.num_special_tokens
        )
        self.track_id_to_artist_id = torch.tensor(self.track_id_to_artist_id, device=device, dtype=torch.long)

        self.track_id_to_artist_id_joint = get_track_id_to_artist_id(
            track2id=self.track2id,
            artist2id=self.artist2id_joint,
            track2artist=self.track2artist,
            num_special_tokens=self.model_joint.num_special_tokens
        )
        self.track_id_to_artist_id_joint = torch.tensor(self.track_id_to_artist_id_joint, device=device, dtype=torch.long)

        self.padding_idx = 0  # TODO
        self.keys_artist = self.model_artist.artist_emb.weight.T  # [D, V_artist]

        n = self.model_joint.num_tracks + self.model_joint.num_special_tokens
        track_ids = torch.arange(n, device=device)
        if self.model_joint.use_artist_emb:
            artist_ids = self.track_id_to_artist_id_joint
        else:
            artist_ids = None
        self.keys_joint = self.model_joint.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids).T  # [D, V_track]

    def __call__(
            self,
            test_data: List[List[str]],
            **kwargs
    ) -> List[List[str]]:
        batch_size = kwargs.pop("batch_size", self.batch_size)
        k = kwargs.pop("k", self.k)
        test = kwargs.pop("test", self.test)

        ds = DatasetV2(
            data=test_data,
            track2id=self.track2id,
            bos_id=self.bos_id,
            unk_id=self.unk_id,
            test=test
        )
        collator = DataCollatorSingleInput(padding_idx=self.padding_idx)
        loader = DataLoader(ds, collate_fn=collator, batch_size=batch_size, shuffle=False)

        res = []
        pbar = tqdm.tqdm(total=len(ds))
        with torch.no_grad():
            for batch in loader:
                track_ids = batch["inputs"].to(self.device)
                batch_size = track_ids.shape[0]

                xs = torch.arange(batch_size, device=self.device)
                padding_mask = get_padding_mask(track_ids, padding_idx=self.padding_idx)  # [N, T]
                ys = padding_mask.sum(1) - 1

                artist_ids = self.track_id_to_artist_id[track_ids]
                outputs = self.model_artist(track_ids=None, artist_ids=artist_ids)  # [N, T, D]
                logits_artist = torch.matmul(outputs[xs, ys, :], self.keys_artist)  # [N, V_artist]

                if self.model_joint.use_artist_emb:
                    artist_ids = self.track_id_to_artist_id_joint[track_ids]
                else:
                    artist_ids = None
                outputs = self.model_joint(track_ids=track_ids, artist_ids=artist_ids)  # [N, T, D]
                logits_joint = torch.matmul(outputs[xs, ys, :], self.keys_joint)  # [N, V_track]

                logits_blend = logits_joint * self.w_joint + logits_artist[:, self.track_id_to_artist_id] * self.w_artist

                # ignore already liked tracks and special tokens
                logits_blend[:, :self.model_joint.num_special_tokens] = -10000.0
                logits_blend[xs[:, None], track_ids] = -10000.0

                preds = (logits_blend * -1.0).argsort(1)[:, :k].to("cpu").numpy()
                for i in range(preds.shape[0]):
                    res_i = []
                    for j in range(preds.shape[1]):
                        res_i.append(self.id2track[preds[i, j]])
                    res.append(res_i)
                pbar.update(batch_size)
        pbar.close()
        return res


class InferenceWrapperCountBasedV1(BaseInferenceWrapper):
    """
    search for similar users with elasticsearch
    """
    def __init__(
            self,
            track_to_artist_path: str,
            track_counts_path: str,
            index: str,
            track_boost: float = 2.0,
            num_users_to_retrieve: int = 100,
            batch_size: int = 64,
            k: int = 100,
            test: bool = False,
            save_candidates_with_scores: bool = False,
            candidates_path: str = None,
            max_candidates_to_save: int = 1000,
            log_rank: bool = False,
            add_artist_clause: bool = True
    ):
        super().__init__(batch_size=batch_size, k=k, test=test)
        self.track_boost = track_boost
        self.num_users_to_retrieve = num_users_to_retrieve
        self.index = index
        if save_candidates_with_scores:
            assert candidates_path is not None
            assert not os.path.exists(candidates_path), f'please rename or remove candidates path: {candidates_path}'
        self.save_candidates_with_scores = save_candidates_with_scores
        self.candidates_path = candidates_path
        self.max_candidates_to_save = max_candidates_to_save
        self.log_rank = log_rank
        self.add_artist_clause = add_artist_clause

        self.es = Elasticsearch()

        self.track2artist = {}
        with open(track_to_artist_path) as f:
            next(f)
            for line in f:
                track, artist = line.strip().split(",")
                self.track2artist[track] = artist

        with open(track_counts_path) as f:
            self.track_counts = json.load(f)

    def __call__(self, test_data: List[List[str]], **kwargs) -> List[List[str]]:
        preds = []
        pbar = tqdm.tqdm(total=len(test_data))
        f = None
        if self.save_candidates_with_scores:
            f = open(self.candidates_path, "w")
        for start in range(0, len(test_data), self.batch_size):
            batch = test_data[start:start + self.batch_size]
            if not self.test:
                batch = [x[:-1] for x in batch]
            res = self._search(batch)
            for i in range(len(batch)):
                history = set(batch[i])
                candidates = defaultdict(float)
                for j, hit in enumerate(res["responses"][i]["hits"]["hits"], 1):
                    for track in hit["_source"]["tracks"]:
                        if track in history:
                            continue
                        if self.log_rank:
                            denominator = math.log(1 + j, 2)
                        else:
                            denominator = j
                        candidates[track] += hit["_score"] / denominator
                candidates = {k: v + math.log(self.track_counts[k]) for k, v in candidates.items()}
                candidates_sorted = sorted(candidates.items(), key=lambda item: -item[1])
                if self.save_candidates_with_scores:
                    f.write(json.dumps(dict(candidates_sorted[:self.max_candidates_to_save])) + "\n")
                preds.append([x[0] for x in candidates_sorted[:self.k]])
                pbar.update(1)
        pbar.close()
        if f is not None:
            f.close()
        return preds

    def _build_query(self, tracks: List[str]):
        tracks_clause = {
            "bool": {"should": [{"term": {"tracks": x}} for x in tracks]}
        }
        if self.add_artist_clause:
            tracks_clause["boost"] = self.track_boost
            artists = {self.track2artist[x] for x in tracks}
            artists_clause = {
                "bool": {"should": [{"term": {"artists": x}} for x in artists]},
            }
            body = {
                "query": {
                    "bool": {
                        "should": [
                            tracks_clause,
                            artists_clause
                        ]
                    }
                },
                "size": self.num_users_to_retrieve,
            }
        else:
            body = {
                "query": {
                    "bool": {
                        "should": tracks_clause
                    }
                },
                "size": self.num_users_to_retrieve,
            }
        return body

    def _search(self, data: List[List[str]]) -> Dict:
        head = {"index": self.index}
        body = []
        for x in data:
            body.append(head)
            body.append(self._build_query(x))
        res = self.es.msearch(body, request_timeout=666666)
        return res


class InferenceWrapperEnsembleV1(BaseInferenceWrapper):
    """
    - 3 track models
    - 3 artist models
    - top-k tracks from es
    - p(track|history) = p(track|artist,history) * p(artist|history)
    - p(track|history) = sum(p(track|track_i) * p(track_i) for track_i in history)
    """
    def __init__(
            self,
            artist_checkpoint_dir_1,
            artist_checkpoint_dir_2,
            artist_checkpoint_dir_3,
            joint_checkpoint_dir_1,
            joint_checkpoint_dir_2,
            joint_checkpoint_dir_3,
            track_to_artist_path: str,
            track_vocab_path: str,
            artist_vocab_path: str,
            artist_vocab_path_joint: str,
            count_features_path: str,
            w_artist_1: float,
            w_artist_2: float,
            w_artist_3: float,
            w_joint_1: float,
            w_joint_2: float,
            w_joint_3: float,
            w_artist: float,
            w_joint: float,
            w_ens: float,
            w_count: float,
            w_probs_ens: float,
            w_probs_count: float,
            w_scores_track_pair: float,
            limit: int = -1,
            bos_id: int = 1,  # warning: bos и unk должны быть одинаковыми у двух моделей
            unk_id: int = 2,
            device: str = "cuda",
            batch_size: int = 32,
            k: int = 100,
            test: bool = False,
            add_probs_count: bool = False,
            track_counts_path: str = None,
            artist_counts_path: str = None,
            top_k_artists: int = 100,
            add_track_pair_counts: bool = False,
            vec_vocab_path: str = None,
            v_path: str = None,
            torch_mm: bool = False
    ):
        if add_probs_count:
            assert track_counts_path is not None
            assert artist_counts_path is not None
        if add_track_pair_counts:
            assert vec_vocab_path is not None
            assert v_path is not None
        device = device if torch.cuda.is_available() else "cpu"
        super().__init__(batch_size=batch_size, k=k, test=test)
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.device = device
        self.add_probs_count = add_probs_count
        self.top_k_artists = top_k_artists
        self.add_track_pair_counts = add_track_pair_counts
        self.torch_mm = torch_mm

        self.w_artist_1 = w_artist_1
        self.w_artist_2 = w_artist_2
        self.w_artist_3 = w_artist_3
        self.w_joint_1 = w_joint_1
        self.w_joint_2 = w_joint_2
        self.w_joint_3 = w_joint_3
        self.w_artist = w_artist
        self.w_joint = w_joint
        self.w_ens = w_ens
        self.w_count = w_count
        self.w_probs_ens = w_probs_ens
        self.w_probs_count = w_probs_count
        self.w_scores_track_pair = w_scores_track_pair

        self.track2artist = {}
        with open(track_to_artist_path) as f:
            next(f)
            for line in f:
                track, artist = line.strip().split(",")
                self.track2artist[track] = artist
        self.artist2tracks = defaultdict(list)
        for track, artist in self.track2artist.items():
            self.artist2tracks[artist].append(track)

        with open(track_vocab_path) as f:
            self.track2id = json.load(f)
        self.id2track = {v: k for k, v in self.track2id.items()}

        with open(artist_vocab_path) as f:
            self.artist2id = json.load(f)
        self.id2artist = {v: k for k, v in self.artist2id.items()}

        with open(artist_vocab_path_joint) as f:
            self.artist2id_joint = json.load(f)

        self.count_feats = []
        with open(count_features_path) as f:
            for line in tqdm.tqdm(f):
                self.count_feats.append(json.loads(line))
                if len(self.count_feats) == limit:
                    break

        self.track_counts = None
        if track_counts_path is not None:
            with open(track_counts_path) as f:
                self.track_counts = json.load(f)

        self.artist_counts = None
        if artist_counts_path is not None:
            with open(artist_counts_path) as f:
                self.artist_counts = json.load(f)

        self.model_artist_1 = load_model(os.path.join(artist_checkpoint_dir_1, ".."), artist_checkpoint_dir_1, device)
        self.model_artist_2 = load_model(os.path.join(artist_checkpoint_dir_2, ".."), artist_checkpoint_dir_2, device)
        self.model_artist_3 = None
        if artist_checkpoint_dir_3 is not None:
            self.model_artist_3 = load_model(os.path.join(artist_checkpoint_dir_3, ".."), artist_checkpoint_dir_3, device)
        self.model_joint_1 = load_model(os.path.join(joint_checkpoint_dir_1, ".."), joint_checkpoint_dir_1, device)
        self.model_joint_2 = load_model(os.path.join(joint_checkpoint_dir_2, ".."), joint_checkpoint_dir_2, device)
        self.model_joint_3 = load_model(os.path.join(joint_checkpoint_dir_3, ".."), joint_checkpoint_dir_3, device)

        self.track_id_to_artist_id = get_track_id_to_artist_id(
            track2id=self.track2id,
            artist2id=self.artist2id,
            track2artist=self.track2artist,
            num_special_tokens=self.model_joint_1.num_special_tokens
        )
        self.track_id_to_artist_id = torch.tensor(self.track_id_to_artist_id, device=device, dtype=torch.long)

        self.track_id_to_artist_id_joint = get_track_id_to_artist_id(
            track2id=self.track2id,
            artist2id=self.artist2id_joint,
            track2artist=self.track2artist,
            num_special_tokens=self.model_joint_1.num_special_tokens
        )
        self.track_id_to_artist_id_joint = torch.tensor(self.track_id_to_artist_id_joint, device=device, dtype=torch.long)

        self.padding_idx = 0  # TODO

        self.keys_artist_1 = self.model_artist_1.artist_emb.weight.T
        self.keys_artist_2 = self.model_artist_2.artist_emb.weight.T
        self.keys_artist_3 = None
        if self.model_artist_3 is not None:
            self.keys_artist_3 = self.model_artist_3.artist_emb.weight.T

        n = self.model_joint_1.num_tracks + self.model_joint_1.num_special_tokens
        track_ids = torch.arange(n, device=device)
        if self.model_joint_1.use_artist_emb:
            artist_ids = self.track_id_to_artist_id_joint
        else:
            artist_ids = None
        self.keys_joint_1 = self.model_joint_1.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids).T

        n = self.model_joint_2.num_tracks + self.model_joint_2.num_special_tokens
        track_ids = torch.arange(n, device=device)
        if self.model_joint_2.use_artist_emb:
            artist_ids = self.track_id_to_artist_id_joint
        else:
            artist_ids = None
        self.keys_joint_2 = self.model_joint_2.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids).T

        n = self.model_joint_3.num_tracks + self.model_joint_3.num_special_tokens
        track_ids = torch.arange(n, device=device)
        if self.model_joint_3.use_artist_emb:
            artist_ids = self.track_id_to_artist_id_joint
        else:
            artist_ids = None
        self.keys_joint_3 = self.model_joint_3.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids).T

        if self.add_track_pair_counts:
            print("loading V matrix...")
            with np.load(v_path) as d:
                self.V = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(d["shape"][0], d["shape"][1]))
                print("shape:", self.V.shape)
                print("nnz:", self.V.nnz)

            if self.torch_mm:
                print(f"converting csr matrix to torch tensor")
                self.V = torch.sparse_csr_tensor(
                    self.V.indptr,
                    self.V.indices,
                    self.V.data,
                    dtype=torch.float32,
                    size=self.V.shape,
                    device="cpu",  # OOM in case of gpu
                    requires_grad=False
                )
                print("done")

            self.vec_vocab = []
            with open(vec_vocab_path) as f:
                for track in f:
                    self.vec_vocab.append(track.strip())
            self.vec = CountVectorizer(binary=True, analyzer=lambda x: x, vocabulary=self.vec_vocab)

    def __call__(
            self,
            test_data: List[List[str]],
            **kwargs
    ) -> List[List[str]]:
        assert len(test_data) <= len(self.count_feats)
        batch_size = kwargs.pop("batch_size", self.batch_size)
        k = kwargs.pop("k", self.k)
        test = kwargs.pop("test", self.test)

        ds = DatasetV2(
            data=test_data,
            track2id=self.track2id,
            bos_id=self.bos_id,
            unk_id=self.unk_id,
            test=test
        )
        collator = DataCollatorSingleInput(padding_idx=self.padding_idx)
        loader = DataLoader(ds, collate_fn=collator, batch_size=batch_size, shuffle=False)

        res = []
        offset = 0
        with torch.no_grad(), tqdm.tqdm(total=len(ds), position=0, leave=True) as pbar:
            for batch in loader:
                track_ids = batch["inputs"].to(self.device)
                batch_size = track_ids.shape[0]

                xs = torch.arange(batch_size, device=self.device)
                padding_mask = get_padding_mask(track_ids, padding_idx=self.padding_idx)  # [N, T]
                ys = padding_mask.sum(1) - 1

                # artist
                artist_ids = self.track_id_to_artist_id[track_ids]
                outputs = self.model_artist_1(track_ids=None, artist_ids=artist_ids)  # [N, T, D]
                logits_artist_1 = torch.matmul(outputs[xs, ys, :], self.keys_artist_1)  # [N, V_artist]
                outputs = self.model_artist_2(track_ids=None, artist_ids=artist_ids)  # [N, T, D]
                logits_artist_2 = torch.matmul(outputs[xs, ys, :], self.keys_artist_2)  # [N, V_artist]
                logits_artist = logits_artist_1 * self.w_artist_1 + logits_artist_2 * self.w_artist_2
                if self.model_artist_3 is not None:
                    outputs = self.model_artist_3(track_ids=None, artist_ids=artist_ids)  # [N, T, D]
                    logits_artist_3 = torch.matmul(outputs[xs, ys, :], self.keys_artist_3)  # [N, V_artist]
                    logits_artist += logits_artist_3 * self.w_artist_3

                # joint
                if self.model_joint_1.use_artist_emb:
                    artist_ids = self.track_id_to_artist_id_joint[track_ids]
                else:
                    artist_ids = None
                outputs = self.model_joint_1(track_ids=track_ids, artist_ids=artist_ids)  # [N, T, D]
                logits_joint_1 = torch.matmul(outputs[xs, ys, :], self.keys_joint_1)  # [N, V_track]

                if self.model_joint_2.use_artist_emb:
                    artist_ids = self.track_id_to_artist_id_joint[track_ids]
                else:
                    artist_ids = None
                outputs = self.model_joint_2(track_ids=track_ids, artist_ids=artist_ids)  # [N, T, D]
                logits_joint_2 = torch.matmul(outputs[xs, ys, :], self.keys_joint_2)  # [N, V_track]

                if self.model_joint_3.use_artist_emb:
                    artist_ids = self.track_id_to_artist_id_joint[track_ids]
                else:
                    artist_ids = None
                outputs = self.model_joint_3(track_ids=track_ids, artist_ids=artist_ids)  # [N, T, D]
                logits_joint_3 = torch.matmul(outputs[xs, ys, :], self.keys_joint_3)  # [N, V_track]

                logits_joint = logits_joint_1 * self.w_joint_1 + logits_joint_2 * self.w_joint_2 + logits_joint_3 * self.w_joint_3

                # joint + artist
                logits_ens = logits_joint * self.w_joint + logits_artist[:, self.track_id_to_artist_id] * self.w_artist

                logits_counts = self._build_counts_dense_matrix(offset, batch_size)
                logits = logits_ens * self.w_ens + logits_counts * self.w_count

                if self.add_probs_count:
                    probs_ens = torch.softmax(logits, dim=-1)
                    probs_count = self._get_probs_count(
                        logits_artist=logits_artist,
                        test_data=test_data,
                        offset=offset,
                        test=test,
                        num_tracks=logits.shape[1]
                    )
                    scores = probs_ens * self.w_probs_ens + probs_count * self.w_probs_count
                else:
                    scores = logits

                if self.add_track_pair_counts:
                    if test:
                        X_test = self.vec.transform(test_data[offset:offset + batch_size])
                    else:
                        X_test = self.vec.transform([x[:-1] for x in test_data[offset:offset + batch_size]])
                    X_test = X_test.astype(float).tolil()
                    for i in range(X_test.shape[0]):
                        if test:
                            history = test_data[offset + i]
                        else:
                            history = test_data[offset + i][:-1]
                        n = len(history)
                        for j in range(n):
                            track_id = self.vec.vocabulary_.get(history[j])
                            if track_id is not None:
                                X_test[i, track_id] = 1 / (n - j)
                    X_test = X_test.tocsr()
                    if self.torch_mm:
                        X_test = torch.sparse_csr_tensor(
                            X_test.indptr,
                            X_test.indices,
                            X_test.data,
                            dtype=torch.float32,
                            size=X_test.shape,
                            device="cpu",
                            requires_grad=False
                        )
                        Y = torch.sparse.mm(X_test, self.V)
                        Y = Y.to_dense().to(self.device)
                    else:
                        Y = X_test.dot(self.V)
                        Y = torch.tensor(Y.A, device=scores.device, dtype=scores.dtype)
                    scores_track_pair = torch.zeros_like(scores)
                    ys = torch.tensor([self.track2id[track] for track in self.vec_vocab], device=scores.device)
                    scores_track_pair[:, ys] = Y
                    scores += scores_track_pair * self.w_scores_track_pair

                # ignore already liked tracks and special tokens
                scores[:, :self.model_joint_1.num_special_tokens] = -10000.0
                scores[xs[:, None], track_ids] = -10000.0

                preds = (scores * -1.0).argsort(1)[:, :k]
                for i in range(preds.shape[0]):
                    res_i = []
                    for j in range(preds.shape[1]):
                        res_i.append(self.id2track[preds[i, j].item()])
                    res.append(res_i)

                pbar.update(batch_size)
                offset += batch_size
        return res

    def _build_counts_dense_matrix(self, offset: int, n: int) -> torch.Tensor:
        logits_counts = torch.zeros((n, self.keys_joint_1.shape[1]), device=self.device)
        xs = []
        ys = []
        values = []
        for i in range(n):
            counts_i = self.count_feats[offset + i]
            m = 1e10
            for track, score in counts_i.items():
                xs.append(i)
                ys.append(self.track2id[track])
                values.append(score)
                if score < m:
                    m = score
            logits_counts[i] = m
        logits_counts[xs, ys] = torch.tensor(values, device=self.device)
        return logits_counts

    def _get_probs_count(
            self, logits_artist: torch.Tensor, test_data: List[List[str]], offset: int, test: bool, num_tracks: int,
    ) -> torch.Tensor:
        probs = torch.zeros((logits_artist.shape[0], num_tracks), device=logits_artist.device)
        probs_artist = torch.softmax(logits_artist, dim=-1)
        topk = torch.topk(probs_artist, k=self.top_k_artists, dim=-1)
        for i in range(probs_artist.shape[0]):
            if test:
                history_set = set(test_data[offset + i])
            else:
                history_set = set(test_data[offset + i][:-1])
            tracks = []
            scores = []
            for j in range(self.top_k_artists):
                artist_id = topk.indices[i, j].item()
                p_artist = topk.values[i, j].item()
                artist = self.id2artist[artist_id]
                artist_counts_i = self.artist_counts[artist]
                new_tracks = []
                for track in self.artist2tracks[artist]:
                    if track in history_set:
                        artist_counts_i -= self.track_counts[track]
                    else:
                        new_tracks.append(track)
                for track in new_tracks:
                    p_track = self.track_counts[track] / artist_counts_i
                    score = p_artist * p_track
                    tracks.append(track)
                    scores.append(score)
            probs[i, [self.track2id[track] for track in tracks]] = torch.tensor(scores, device=probs.device)
        return probs
