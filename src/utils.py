from typing import List, Dict


def get_track_id_to_artist_id(
        track2id: Dict[str, int],
        artist2id: Dict[str, int],
        track2artist: Dict[str, str],
        num_special_tokens: int
) -> List[int]:
    n = max(track2id.values()) + 1
    assert n == len(track2id) + num_special_tokens, f'{n} != {len(track2id) + num_special_tokens}'
    track_id_to_artist_id = [0] * n
    for i in range(num_special_tokens):
        track_id_to_artist_id[i] = i
    for track, artist in track2artist.items():
        track_id = track2id[track]
        assert track_id >= num_special_tokens, f'{track_id} < {num_special_tokens}'
        artist_id = artist2id[artist]
        track_id_to_artist_id[track_id] = artist_id
    return track_id_to_artist_id
