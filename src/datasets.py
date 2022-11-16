import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset


# * document = track_emb + artist_emb
# * custom negatives for each timestamp

class TrainingDatasetV1(Dataset):
    def __init__(
            self,
            data,
            track2id,
            artist2id,
            track2artist,
            num_negatives: int = 100,
            bos_id: int = 1,
            unk_id: int = 2  # not used here yet, but shared with valid and test
    ):
        super().__init__()
        self.data = data
        self.track2id = track2id
        self.artist2id = artist2id
        self.track2artist = track2artist
        self.num_negatives = num_negatives
        self.bos_id = bos_id
        self.unk_id = unk_id

        self.tracks = None

    def set_tracks(self):
        self.tracks = list(self.track2id.keys())

    def __getitem__(self, index: int):
        if self.tracks is None:
            self.set_tracks()

        history = self.data[index]  # [T]

        # randomly sample songs, which user didn't like
        negatives = self.sample_negatives(history)

        tracks = history + negatives
        track_ids = []
        artist_ids = []
        for track in tracks:
            # assume that during training we might also don't know each track (e.g. when truncating rare tracks)
            track_id = self.track2id.get(track, self.unk_id)
            track_ids.append(track_id)
            # artist is always known
            artist = self.track2artist[track]
            artist_ids.append(self.artist2id[artist])

        # pack all ids (length * (2 + num_negatives)): for each timestamp we need:
        # * one input element
        # * ont target element
        # * `num_negatives` negative samples
        # input = bos + history without last element
        # targets = history
        length = len(history)
        track_ids = [self.bos_id] + track_ids[:length - 1] + track_ids[:length] + track_ids[length:]
        artist_ids = [self.bos_id] + artist_ids[:length - 1] + artist_ids[:length] + artist_ids[length:]

        track_ids = torch.tensor(track_ids).reshape(-1, length).T  # [M, T] -> [T, M],  M = num_negatives + 2
        artist_ids = torch.tensor(artist_ids).reshape(-1, length).T
        return track_ids, artist_ids

    def __len__(self):
        return len(self.data)

    def sample_negatives(self, history: List[int]) -> List[int]:
        length = len(history)
        history_set = set(history)
        negatives = set()
        sample_limit = min(length * self.num_negatives, len(self.tracks) - length)  # to prevent infinite loop
        while len(negatives) != sample_limit:
            i = random.choice(self.tracks)
            if (i in history_set) or (i in negatives):
                continue
            negatives.add(i)
        negatives = list(negatives)
        while len(negatives) != length * self.num_negatives:
            negatives.append(negatives[-1])
        return negatives


class ValidDatasetV1(Dataset):
    def __init__(
            self,
            data,
            track2id,
            artist2id,
            track2artist,
            bos_id: int = 1,
            unk_id: int = 2,
    ):
        super().__init__()
        self.data = data
        self.track2id = track2id
        self.artist2id = artist2id
        self.track2artist = track2artist
        self.bos_id = bos_id
        self.unk_id = unk_id

    def __getitem__(self, index: int):
        history = self.data[index]
        track_ids = []
        artist_ids = []
        for track in history:
            # track might be new
            track_id = self.track2id.get(track, self.unk_id)
            track_ids.append(track_id)
            # artist is known for each track
            artist = self.track2artist[track]
            artist_ids.append(self.artist2id[artist])

        # concat x and y like in training
        track_ids = [self.bos_id] + track_ids[:-1] + track_ids
        artist_ids = [self.bos_id] + artist_ids[:-1] + artist_ids

        # pack elements in training format
        length = len(history)
        track_ids = torch.tensor(track_ids).reshape(-1, length).T
        artist_ids = torch.tensor(artist_ids).reshape(-1, length).T
        return track_ids, artist_ids

    def __len__(self):
        return len(self.data)


class TestDatasetV1(Dataset):
    def __init__(
            self,
            data,
            track2id,
            artist2id,
            track2artist,
            bos_id: int = 1,
            unk_id: int = 2,
    ):
        super().__init__()
        self.data = data
        self.track2id = track2id
        self.artist2id = artist2id
        self.track2artist = track2artist
        self.bos_id = bos_id
        self.unk_id = unk_id

    def __getitem__(self, index: int):
        history = self.data[index]
        track_ids = []
        artist_ids = []
        for track in history:
            # track might be new
            track_id = self.track2id.get(track, self.unk_id)
            track_ids.append(track_id)
            # artist is known for each track
            artist = self.track2artist[track]
            artist_ids.append(self.artist2id[artist])

        track_ids = [self.bos_id] + track_ids
        artist_ids = [self.bos_id] + artist_ids

        # pack elements in training format
        track_ids = torch.tensor(track_ids).unsqueeze(1)  # [T, 1]
        artist_ids = torch.tensor(artist_ids).unsqueeze(1)
        return track_ids, artist_ids

    def __len__(self):
        return len(self.data)


# * document = track_emb + artist_emb
# * artist_id inferred by track_id by lookup in array track_id_to_artist_id
# * custom negatives are generated with torch.randint


class DatasetV2(Dataset):
    """
    track_ids only
    """
    def __init__(
            self,
            data,
            track2id,
            bos_id: int = 1,
            unk_id: int = 2,
            test: bool = False,
            **kwargs
    ):
        super().__init__()
        self.data = data
        self.track2id = track2id
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.test = test

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        history = self.data[index]  # [T]

        track_ids = []
        for track in history:
            track_id = self.track2id.get(track, self.unk_id)
            track_ids.append(track_id)

        if self.test:
            track_ids_x = [self.bos_id] + track_ids
            track_ids_y = [self.bos_id] + track_ids
        else:
            track_ids_x = [self.bos_id] + track_ids[:- 1]
            track_ids_y = track_ids

        return track_ids_x, track_ids_y

    def __len__(self):
        return len(self.data)


# * document = artist_emb
# * p(track|history) = p(track|artist, history) * p(artist|history)
#       p(artist|history) - fitted by model
#       p(track|artist, history) - estimated by count(track) / count(artist),
#       where count(artist) = sum(count(track) for track in tracks(artist) if track not in history)


class DatasetV3(Dataset):
    """
    artist_ids only
    """
    def __init__(
            self,
            data,
            track2artist,
            artist2id,
            bos_id: int = 1,
            unk_id: int = 2,
            test: bool = False,
            **kwargs
    ):
        super().__init__()
        self.data = data
        self.track2artist = track2artist
        self.artist2id = artist2id
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.test = test

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        history = self.data[index]  # [T]

        artist_ids = []
        for track in history:
            artist = self.track2artist[track]
            artist_id = self.artist2id[artist]
            artist_ids.append(artist_id)

        if self.test:
            artist_ids_x = [self.bos_id] + artist_ids
            artist_ids_y = [self.bos_id] + artist_ids
        else:
            artist_ids_x = [self.bos_id] + artist_ids[:- 1]
            artist_ids_y = artist_ids

        return artist_ids_x, artist_ids_y

    def __len__(self):
        return len(self.data)


# * document = track_emb + artist_emb
# * model is learned on two targets: tracks and artists

class DatasetV4(Dataset):
    """
    track_ids, artist_ids
    """
    def __init__(
            self,
            data,
            track2artist,
            track2id,
            artist2id,
            bos_id: int = 1,
            unk_id: int = 2,
            test: bool = False,
            **kwargs
    ):
        super().__init__()
        self.data = data
        self.track2artist = track2artist
        self.track2id = track2id
        self.artist2id = artist2id
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.test = test

    def __getitem__(self, index: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        history = self.data[index]  # [T]
        track_ids = []
        artist_ids = []
        for track in history:
            # track might be new
            track_id = self.track2id.get(track, self.unk_id)
            track_ids.append(track_id)
            # artist is known for each track
            artist = self.track2artist[track]
            artist_ids.append(self.artist2id[artist])

        if self.test:
            track_ids_x = [self.bos_id] + track_ids
            track_ids_y = [self.bos_id] + track_ids
            artist_ids_x = [self.bos_id] + artist_ids
            artist_ids_y = [self.bos_id] + artist_ids
        else:
            track_ids_x = [self.bos_id] + track_ids[:-1]
            track_ids_y = track_ids
            artist_ids_x = [self.bos_id] + artist_ids[:-1]
            artist_ids_y = artist_ids
        return track_ids_x, track_ids_y, artist_ids_x, artist_ids_y

    def __len__(self):
        return len(self.data)
