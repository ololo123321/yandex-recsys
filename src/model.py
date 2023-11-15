import torch
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.activations import NewGELUActivation


T = torch.Tensor


class AttentionTypes:
    DOT_PRODUCT = "dot_product"
    RELATIVE = "relative"


class Activations:
    RELU = "relu"
    GeluNew = "gelu_new"


class EmbeddingWeightingTypes:
    AVERAGE = "average"
    SIGMOID = "sigmoid"


class Decoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            num_heads: int = 8,
            head_dim: int = 32,
            dff: int = 1024,
            dropout: float = 0.1,
            max_length: int = 512,  # for positional embeddings and relative attention
            padding_idx: int = 0,
            num_special_tokens: int = 3,  # pad, bos, unk
            use_pos_emb: bool = True,
            use_track_emb: bool = True,
            use_artist_emb: bool = True,
            num_tracks: int = None,
            num_artists: int = None,
            attention_type: str = AttentionTypes.DOT_PRODUCT,  # {dot_product, relative}
            activation: str = Activations.RELU,  # {relu, gelu_new}
            emb_weighting_type: str = EmbeddingWeightingTypes.AVERAGE,  # {average, sigmoid}
            **kwargs
    ):
        assert attention_type in {AttentionTypes.DOT_PRODUCT, AttentionTypes.RELATIVE}, attention_type
        assert activation in {Activations.RELU, Activations.GeluNew}, activation
        assert emb_weighting_type in {EmbeddingWeightingTypes.AVERAGE, EmbeddingWeightingTypes.SIGMOID}

        super().__init__()

        assert use_track_emb or use_artist_emb
        self.num_tracks = num_tracks
        self.num_artists = num_artists
        self.d_model = num_heads * head_dim
        self.dropout = dropout

        self.padding_idx = padding_idx
        self.num_special_tokens = num_special_tokens
        self.use_pos_emb = use_pos_emb
        self.use_track_emb = use_track_emb
        self.use_artist_emb = use_artist_emb
        self.emb_weighting_type = emb_weighting_type

        assert (num_tracks is not None) or (num_artists is not None)

        self.track_emb = None
        if self.use_track_emb:
            assert num_tracks is not None
            n = num_tracks + num_special_tokens
            self.track_emb = nn.Embedding(n, self.d_model, _weight=self.get_init_emb_matrix(n))

        self.artist_emb = None
        if self.use_artist_emb:
            assert num_artists is not None
            n = num_artists + num_special_tokens
            self.artist_emb = nn.Embedding(n, self.d_model, _weight=self.get_init_emb_matrix(n))

        self.pos_emb = None
        if use_pos_emb:
            # no need to custom init, because it doesn't participate in final logits
            self.pos_emb = nn.Embedding(max_length, self.d_model)
            self.positions = torch.arange(max_length)[None]  # [1, T]
        self.emb_dropout = nn.Dropout(dropout)

        self.w_track_emb = None
        self.emb_sigmoid = None
        if self.emb_weighting_type == "sigmoid":
            self.w_track_emb = nn.Parameter(torch.zeros(()))
            self.emb_sigmoid = nn.Sigmoid()

        self.dec_layers = nn.ModuleList([
            DecoderLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                dff=dff,
                dropout=dropout,
                attention_type=attention_type,
                maxlen=max_length,
                activation=activation
            )
            for _ in range(num_layers)
        ])

    def forward(self, track_ids: T = None, artist_ids: T = None) -> T:
        """
        track_ids: [N, T], int.
        artist_ids: [N, T], int

        :return:
        logits: [N, T, D], float
        """
        assert (track_ids is not None) or (artist_ids is not None)
        x = self.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids)
        if self.pos_emb is not None:
            x += self.pos_emb(self.positions[:, :x.shape[1]].to(x.device))
        x = self.emb_dropout(x)

        ids = track_ids if track_ids is not None else artist_ids
        padding_mask = get_padding_mask(ids, self.padding_idx).to(x.device)  # [N, T]
        causal_mask = get_casual_mask(ids.shape[1]).to(x.device)  # [T, T]
        mask = torch.logical_and(causal_mask[None, None, :, :], padding_mask[:, None, None, :])  # [N, 1, T, T]
        mask = mask.float()

        for dec in self.dec_layers:
            x = dec(x, mask=mask)  # [N, T, D]
        return x

    def get_items_embeddings(self, track_ids: T = None, artist_ids: T = None) -> T:
        assert (track_ids is not None) or (artist_ids is not None)
        if track_ids is not None:
            assert self.track_emb is not None
            x = self.track_emb(track_ids)
            if artist_ids is not None:
                assert self.artist_emb is not None
                if self.emb_weighting_type == "average":
                    x += self.artist_emb(artist_ids)
                    x *= 0.5
                elif self.emb_weighting_type == "sigmoid":
                    w = self.emb_sigmoid(self.w_track_emb)
                    x = x * w + self.artist_emb(artist_ids) * (1.0 - w)
                else:
                    raise NotImplementedError
        else:
            assert self.artist_emb is not None
            x = self.artist_emb(artist_ids)
        return x

    def get_init_emb_matrix(self, vocab_size):
        """
        custom initialization needs due to embedding are used as output layer
        """
        x = torch.zeros((vocab_size, self.d_model))
        return fill_emb_matrix(x, self.d_model)


def get_padding_mask(ids: T, padding_idx: int) -> T:
    return ids.not_equal(padding_idx)


def get_casual_mask(max_length: int) -> T:
    return torch.tril(torch.ones(max_length, max_length)).bool()


def fill_emb_matrix(x: T, dim: int) -> T:
    """
    custom initialization needs due to embedding are used as output layer
    """
    bound = 1.0 / dim ** 0.5
    x.uniform_(-bound, bound)
    return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            num_heads: int = 8,
            head_dim: int = 64,
            dff: int = 1024,
            dropout: float = 0.1,
            attention_type: str = AttentionTypes.DOT_PRODUCT,
            maxlen: int = None,
            activation: str = "relu"
    ):
        super().__init__()
        d_model = num_heads * head_dim

        if attention_type == AttentionTypes.DOT_PRODUCT:
            self.mha = MHA(num_heads, head_dim)
        elif attention_type == "relative":
            self.mha = MHAWithRelativeEmbeddings(num_heads, head_dim, maxlen)
        else:
            raise NotImplementedError(f"expected attention_type in {{dot_product, relative}}, got {attention_type}")

        self.dense_ff = nn.Linear(d_model, dff)

        if activation == "relu":
            self.relu = nn.ReLU()
        elif activation == "gelu_new":
            self.relu = NewGELUActivation()
        else:
            raise NotImplementedError(f"expected activation in {{relu, gelu_new}}, got {activation}")

        self.dense_model = nn.Linear(dff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: T, mask: T = None) -> T:
        y = self.mha(x, mask)
        y = self.dropout1(y)
        x = self.ln1(x + y)
        y = self.dense_ff(x)
        y = self.relu(y)
        y = self.dense_model(y)
        y = self.dropout2(y)
        x = self.ln2(x + y)
        return x


class MHA(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        d_model = num_heads * head_dim
        self.dense_input = nn.Linear(d_model, d_model * 3)

    def forward(self, x: T, mask: T = None) -> T:
        """
        https://arxiv.org/abs/1706.03762
        D = num_heads * head_dim
        :param x: tf.Tensor of shape [N, T, D]
        :param mask: tf.Tensor of shape [N, 1, T, T]. Ones at valid positions
        :return: tf.Tensor of shape [N, T, D]
        """
        batch_size = x.shape[0]
        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = torch.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = torch.permute(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = torch.unbind(qkv, dim=0)  # 3 * [N, H, T, D]

        k = k.permute(0, 1, 3, 2)  # [N, H, D, T]
        logits = torch.matmul(q, k)  # [N, H, T, T]
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        if mask is not None:
            logits += (1.0 - mask) * -10000.0

        w = torch.softmax(logits, dim=-1)  # [N, H, T, T] (k-axis)
        x = torch.matmul(w, v)  # [N, H, T, D]
        x = torch.permute(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = torch.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class MHAWithRelativeEmbeddings(nn.Module):
    def __init__(self, num_heads, head_dim, max_length):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        d_model = num_heads * head_dim
        self.dense_input = nn.Linear(d_model, d_model * 3)

        w = torch.zeros(num_heads, max_length * 2 + 1, head_dim)
        w = fill_emb_matrix(w, head_dim)
        self.R = nn.Parameter(w)

    def forward(self, x: T, mask: T = None) -> T:
        """
        https://arxiv.org/abs/1706.03762
        D = num_heads * head_dim
        :param x: tf.Tensor of shape [N, T, D]
        :param mask: tf.Tensor of shape [N, 1, T, T]. Ones at valid positions
        :return: tf.Tensor of shape [N, T, D]
        """
        batch_size = x.shape[0]
        max_length = x.shape[1]

        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = torch.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = torch.permute(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = torch.unbind(qkv, dim=0)  # 3 * [N, H, T, D]

        dot_product_term = torch.matmul(q, k.transpose(2, 3))  # [N, H, T, T]

        indices_q = torch.arange(x.shape[0])  # [Tq]
        indices_k = torch.arange(x.shape[0])  # [Tk]
        relative_positions = indices_k[None, :] - indices_q[:, None]  # [Tq, Tk]
        # (-inf, inf) -> [-max_length, max_length] -> [0, 2 * max_length]
        relative_positions = torch.clamp(
            relative_positions, -self.max_length, self.max_length
        ) + self.max_length  # [Tq, Tk]
        r = self.R[:, relative_positions, :]  # [H, Tq, Tk, d]
        # [Tq, H, N, d] @ [Tq, H, d, Tk] = [Tq, H, N, Tk] --transpose(0, 2)--> [N, H, Tq, Tk]
        relative_term = torch.matmul(q.permute(2, 1, 0, 3), r.permute(1, 0, 3, 2)).transpose(0, 2)

        logits = dot_product_term + relative_term  # [N, H, T, T]

        logits *= 0.5
        logits *= 1.0 / self.head_dim ** 0.5  # [N, H, T, T]

        if mask is not None:
            logits += (1.0 - mask) * -10000.0

        w = torch.softmax(logits, dim=-1)  # [N, H, T, T] (k-axis)
        x = torch.matmul(w, v)  # [N, H, T, D]
        x = torch.permute(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = torch.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class GPT2Artist(nn.Module):
    """
    for convergence comparison (sanity check)
    """
    def __init__(
            self,
            num_layers: int = 6,
            num_heads: int = 8,
            head_dim: int = 32,
            dropout: float = 0.1,
            max_length: int = 512,  # for positional embeddings and relative attention
            padding_idx: int = 0,
            num_special_tokens: int = 4,  # pad, bos, unk, eos
            num_artists: int = None,
            use_pos_emb: bool = True,
            **kwargs
    ):
        super().__init__()
        d_model = num_heads * head_dim
        # interesting that there is no configuration for intermediate mlp dim
        n_positions = (max_length if use_pos_emb else 1)
        config = GPT2Config(
            vocab_size=num_artists + num_special_tokens,
            n_positions=n_positions,
            n_embd=d_model,
            n_layer=num_layers,
            n_head=num_heads,
            activation_function="gelu_new",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            scale_attn_weights=True,
            bos_token_id=1,
            eos_token_id=3,  # 2 is already reserved for unk
        )
        self.gpt2 = GPT2Model(config)
        self.padding_idx = padding_idx
        self.artist_emb = self.gpt2.wte  # for interface
        self.use_pos_emb = use_pos_emb
        if not use_pos_emb:
            self.gpt2.wpe.weight.requires_grad = False
            self.gpt2.wpe.weight.zero_()

    # track_ids - for interface
    def forward(self, track_ids: T = None, artist_ids: T = None) -> T:
        padding_mask = get_padding_mask(artist_ids, padding_idx=self.padding_idx)
        position_ids = None
        if not self.use_pos_emb:
            position_ids = torch.zeros_like(artist_ids)
        outputs = self.gpt2(input_ids=artist_ids, attention_mask=padding_mask, position_ids=position_ids)
        return outputs.last_hidden_state


class GPT2Joint(nn.Module):
    """
    for convergence comparison (sanity check)
    """
    def __init__(
            self,
            num_layers: int = 6,
            num_heads: int = 8,
            head_dim: int = 32,
            dff: int = 1024,
            dropout: float = 0.1,
            max_length: int = 512,  # for positional embeddings and relative attention
            padding_idx: int = 0,
            num_special_tokens: int = 3,  # pad, bos, unk
            use_pos_emb: bool = True,
            use_track_emb: bool = True,
            use_artist_emb: bool = True,
            num_tracks: int = None,
            num_artists: int = None,
            **kwargs
    ):
        super().__init__()
        d_model = num_heads * head_dim

        self.use_track_emb = use_track_emb
        self.use_artist_emb = use_artist_emb
        vocab_size = num_special_tokens
        if self.use_artist_emb:
            assert num_artists is not None
            vocab_size += num_artists
        if self.use_track_emb:
            assert num_tracks is not None
            vocab_size += num_tracks

        # interesting that there is no configuration for intermediate mlp dim
        n_positions = (max_length if use_pos_emb else 1)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=d_model,
            n_layer=num_layers,
            n_head=num_heads,
            activation_function="gelu_new",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            scale_attn_weights=True,
            bos_token_id=1,
            eos_token_id=3,  # 2 is already reserved for unk
        )
        self.gpt2 = GPT2Model(config)
        self.padding_idx = padding_idx

        # for interface
        self.track_emb = None
        if self.use_track_emb:
            self.track_emb = self.gpt2.wte
        self.artist_emb = None
        if self.use_artist_emb:
            self.artist_emb = self.gpt2.wte

        self.num_tracks = num_tracks
        self.num_artists = num_artists
        self.num_special_tokens = num_special_tokens
        self.use_pos_emb = use_pos_emb

        if not use_pos_emb:
            self.gpt2.wpe.weight.requires_grad = False
            self.gpt2.wpe.weight.zero_()

    def forward(self, track_ids: T = None, artist_ids: T = None) -> T:
        if track_ids is not None:
            assert self.use_track_emb
            position_ids = None
            if not self.use_pos_emb:
                position_ids = torch.zeros_like(track_ids)
            padding_mask = get_padding_mask(track_ids, padding_idx=self.padding_idx)
            outputs = self.gpt2(
                input_ids=track_ids, token_type_ids=artist_ids, attention_mask=padding_mask, position_ids=position_ids
            )
        else:
            assert self.use_artist_emb
            position_ids = None
            if not self.use_pos_emb:
                position_ids = torch.zeros_like(artist_ids)
            padding_mask = get_padding_mask(artist_ids, padding_idx=self.padding_idx)
            outputs = self.gpt2(
                input_ids=artist_ids, token_type_ids=None, attention_mask=padding_mask, position_ids=position_ids
            )
        return outputs.last_hidden_state

    def get_items_embeddings(self, track_ids: T = None, artist_ids: T = None) -> T:
        assert (track_ids is not None) or (artist_ids is not None)
        if track_ids is not None:
            assert self.use_track_emb
            x = self.track_emb(track_ids)
            if artist_ids is not None:
                assert self.use_artist_emb
                x += self.artist_emb(artist_ids)
                x *= 0.5  # average
        else:
            assert self.use_artist_emb
            x = self.artist_emb(artist_ids)
        return x
