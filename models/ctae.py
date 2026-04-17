import math
import torch
import torch.nn as nn
from typing import List, Dict


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding added to input sequences.

    Args:
        d_model (int): Feature dimension.
        max_len (int): Maximum sequence length.

    Input:
        x: (T, B, d_model)

    Output:
        x with positional encoding added (same shape).
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding added to input sequences.

    Args:
        d_model (int): Feature dimension.
        max_len (int): Maximum supported sequence length.

    Input:
        x: Tensor of shape (T, B, d_model).

    Output:
        Tensor of shape (T, B, d_model) with learned positional embeddings added.
    """
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0), :].unsqueeze(1)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder that maps input sequences to latent representations.

    Args:
        input_dim (int): Input feature dimension.
        latent_dim (int): Output latent dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_len (int): Maximum supported sequence length.
        pe (bool): Whether to add positional encoding to the decoder input.
        pe_learn (bool): If True, use learnable positional encoding; otherwise use sinusoidal encoding.

    Input:
        src: (T, B, input_dim)

    Output:
        memory: (T, B, latent_dim)
    """
    def __init__(self, input_dim, latent_dim, nhead, num_layers, max_len=5000, pe=False, pe_learn=False, batch_norm=False, causal=True):
        super(TransformerEncoder, self).__init__()
        self.pe = pe
        self.pe_learn = pe_learn
        self.batch_norm = batch_norm
        self.causal = causal
        if pe and not pe_learn:
            self.positional_encoding = PositionalEncoding(input_dim, max_len)
        elif pe and pe_learn:
            self.positional_encoding = LearnablePositionalEncoding(input_dim, max_len)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, latent_dim)
        self.batch_norm = nn.BatchNorm1d(latent_dim, affine=False)

        if self.causal:
            mask = torch.triu(torch.full((max_len, max_len),
                                        float('-inf')), diagonal=1)
            self.register_buffer("causal_mask", mask)

    def forward(self, src): #batch_norm
        if self.pe:
            src = self.positional_encoding(src)

        if self.causal:
            # slice the pre-built mask to the current sequence length
            T = src.size(0)
            mask = self.causal_mask[:T, :T]
            memory = self.transformer_encoder(src, mask=mask)
        else:
            memory = self.transformer_encoder(src)
        memory = self.fc(memory)

        if self.batch_norm:
            memory = memory.permute(1, 2, 0)  # (batch_size, latent_dim, num_timeframes)
            memory = self.batch_norm(memory)
            memory = memory.permute(2, 0, 1)  # (num_timeframes, batch_size, latent_dim)

        return memory
    
    

class TransformerDecoder(nn.Module):
    """
    Transformer decoder that maps latent sequences back to observation space.

    Args:
        latent_dim (int): Latent feature dimension.
        output_dim (int): Output feature dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer decoder layers.
        max_len (int): Maximum supported sequence length.
        pe (bool): Whether to add positional encoding to the decoder input.
        pe_learn (bool): If True, use learnable positional encoding; otherwise use sinusoidal encoding.

    Input:
        tgt: Tensor of shape (T, B, latent_dim), typically initialized as zeros.
        memory: Tensor of shape (T, B, latent_dim) containing latent representations to decode from.

    Output:
        Tensor of shape (T, B, output_dim).
    """
    def __init__(self, latent_dim, output_dim, nhead, num_layers, max_len=5000, pe=False, pe_learn=False,causal=True):
        super(TransformerDecoder, self).__init__()
        self.pe = pe
        self.pe_learn = pe_learn
        self.causal = causal
        if (pe and (not pe_learn)):
            self.positional_encoding = PositionalEncoding(latent_dim, max_len)
        elif pe and pe_learn:
            self.positional_encoding = LearnablePositionalEncoding(latent_dim, max_len)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(latent_dim, output_dim)

        if self.causal:
            mask = torch.triu(torch.full((max_len, max_len),
                                        float('-inf')), diagonal=1)
            self.register_buffer("causal_mask", mask)

    def forward(self, tgt, memory):
        if self.pe:
            tgt = self.positional_encoding(tgt)
        if self.causal:
            T = tgt.size(0)
            mask = self.causal_mask[:T, :T]
            output = self.transformer_decoder(tgt, memory, tgt_mask=mask)
        else:
            output = self.transformer_decoder(tgt, memory)
        output = self.fc(output)
        return output


def build_latent_layout_and_masks(dimension: Dict[str, int], N: int, segment_order=None):
    """
    Construct latent subspace layout and masks from bitstring specification.

    Args:
        dimension: dict mapping bitstring (e.g., '110') → latent dimension.
                   Each bit indicates which regions participate in that subspace.
        N: number of regions.
        segment_order: optional ordering of subspaces.

    Returns:
        segments: list of (bitstring, (start, end)) defining latent partitions.
        D: total latent dimension.
        region_masks: list of N tensors of shape (D,) indicating participation per region.
        subset_masks: dict mapping bitstring → (D,) mask for that subspace.
    """

    def popcount(s): return s.count('1')
    
    if segment_order is None:
        ordered_keys = sorted(dimension.keys(), key=lambda k: (popcount(k), k))
    else:
        ordered_keys = [k for k in segment_order if k in dimension and dimension[k] > 0]

    segments = []
    cursor = 0
    for k in ordered_keys:
        size = int(dimension[k])
        if size <= 0: 
            continue
        segments.append((k, (cursor, cursor + size)))
        cursor += size
    D = cursor

    # region masks
    region_masks = []
    for r in range(N):
        mask = torch.zeros(D, dtype=torch.float32)
        for k, (a, b) in segments:
            if k[r] == '1':  # region r participates in this subspace
                mask[a:b] = 1.0
        region_masks.append(mask)

    # exact-subset masks
    subset_masks = {}
    for k, (a, b) in segments:
        m = torch.zeros(D, dtype=torch.float32)
        m[a:b] = 1.0
        subset_masks[k] = m

    return segments, D, region_masks, subset_masks


class CoupledTransformerAutoencoderTwoRegions(nn.Module):
    def __init__(self, 
                    input_dim1, 
                    input_dim2, 
                    r1_specific_dim, 
                    r2_specific_dim, 
                    shared_dim,
                    nhead=2, 
                    num_layers=2, 
                    num_layers2=None,
                    max_len=5000, 
                    pe=False, pe_learn=False, 
                    batch_norm=True):
        super(CoupledTransformerAutoencoderTwoRegions, self).__init__()

        self.latent_dim1 = shared_dim+r1_specific_dim
        self.latent_dim2 = shared_dim+r2_specific_dim
        self.latent_dim = shared_dim+r1_specific_dim+r2_specific_dim
        self.shared_dim = shared_dim
        self.r1_specific_dim = r1_specific_dim
        self.r2_specific_dim = r2_specific_dim

        if num_layers2 is None:
            num_layers2 = num_layers
#         self.shared_latent_dim = shared_latent_dim

        self.encoder1 = TransformerEncoder(input_dim1, self.latent_dim, nhead, num_layers, max_len, pe, pe_learn, batch_norm)
        self.encoder2 = TransformerEncoder(input_dim2, self.latent_dim, nhead, num_layers2, max_len, pe, pe_learn, batch_norm)
        self.decoder1 = TransformerDecoder(self.latent_dim, input_dim1, nhead, num_layers, max_len, pe, pe_learn)
        self.decoder2 = TransformerDecoder(self.latent_dim, input_dim2, nhead, num_layers2, max_len, pe, pe_learn)

        self.linear_shared_1 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.linear_shared_2 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        self.weights1 = [1 for _ in range(shared_dim)]+[1 for _ in range(r1_specific_dim)]+[0 for _ in range(r2_specific_dim)]
        self.weights2 = [1 for _ in range(shared_dim)]+[0 for _ in range(r1_specific_dim)]+[1 for _ in range(r2_specific_dim)]


    def split_data(self, x, num_neurons1=None):
        x = x.permute(1, 0, 2)  # (num_timeframes, batch_size, input_dim)
        # Here, the dimensions of x: (#time_points, #neurons)
        if num_neurons1 is None:
            num_neurons1 = x.shape[-1]//2
        x1 = x[:, :, :num_neurons1]
        x2 = x[:, :, num_neurons1:]
        return x1, x2


    def forward(self, x, num_neurons1=None):
        x1, x2 = self.split_data(x, num_neurons1=num_neurons1)

        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        # Projecting the latent spaces into shared subspace
        z1_full = self.linear_shared_1(z1)
        z2_full = self.linear_shared_2(z2)


        D = self.latent_dim
        self.weights_tensor1 = torch.tensor(self.weights1, dtype=z1_full.dtype, device=z1_full.device)
        # Reshape/unsqueeze to (1, 1, D) so it can broadcast across (B, T, D)
        self.weights_tensor1 = self.weights_tensor1.view(1, 1, D) 

        self.weights_tensor2 = torch.tensor(self.weights2, dtype=z1_full.dtype, device=z1_full.device)
        # Reshape/unsqueeze to (1, 1, D) so it can broadcast across (B, T, D)
        self.weights_tensor2 = self.weights_tensor2.view(1, 1, D)

        self.shared_mask_tensor = ((self.weights_tensor1 == 1) & (self.weights_tensor2 == 1)).float()


        z = ((z1_full*self.weights_tensor1) + (z2_full*self.weights_tensor2))/(self.weights_tensor1+self.weights_tensor2)#(B, T, latent_dim)

        shared_subspace1 = z1_full[:,:,:self.shared_dim]
        shared_subspace2 = z2_full[:,:,:self.shared_dim]
        specific_subspace1 = z1_full[:,:,self.shared_dim:self.shared_dim+self.r1_specific_dim]
        specific_subspace2 = z2_full[:,:,self.shared_dim+self.r1_specific_dim:]

        shared_subspace = (shared_subspace1+shared_subspace2)/2

        decoder_input11 = z * self.weights_tensor1
        decoder_input22 = z * self.weights_tensor2
        z_shared = z * self.shared_mask_tensor

        # Initialize decoder input (tgt) as zeros
        tgt1 = torch.zeros_like(decoder_input11)
        tgt2 = torch.zeros_like(decoder_input22)

        # Decoding using the shared subspaces
        x11_hat = self.decoder1(tgt1,decoder_input11)
        x12_hat = self.decoder2(tgt2,z_shared)
        x22_hat = self.decoder2(tgt2,decoder_input22)
        x21_hat = self.decoder1(tgt1,z_shared)

        return x11_hat, x22_hat, x12_hat, x21_hat,shared_subspace1,shared_subspace2,specific_subspace1,specific_subspace2,z

    def get_latent_repr(self, x):
        x1, x2 = self.split_data(x)
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        return z1, z2

    def get_shared_latent_repr(self, x):
        z1, z2 = self.get_latent_repr(x)
        s1 = self.linear1(z1)
        s2 = self.linear2(z2)
        return s1, s2

    def get_latent_repr_array(self, x):
        z1, z2 = self.get_latent_repr(x)
        return z1.detach().cpu().numpy(), z2.detach().cpu().numpy()

    def get_shared_repr_array(self, x):
        s1, s2 = self.get_shared_latent_repr(x)
        return s1.detach().cpu().numpy(), s2.detach().cpu().numpy()

    def get_recons(self, x, id=11):
        out = self.forward(x)
        return out

    def get_recons_array(self, x, id=11):
        x_pred = self.get_recons(x, id=id)
        x_pred_list = [i.detach().cpu().numpy() for i in x_pred]
        return x_pred_list



class CoupledTransformerAutoencoderMultiRegion(nn.Module):
    """
    Coupled transformer autoencoder for multi-region neural data.

    This model learns shared and region-specific latent subspaces defined by
    bitstring masks in `dimension`. Each region is encoded independently into a
    common latent space, fused across regions using masked averaging, and then
    decoded back into each region's observation space.

    Args:
        input_dim (List[int]): Input dimension for each region.
        dimension (Dict[str, int]): Mapping from bitstring subspace labels to latent dimensions.
            Each bitstring has length N, where N is the number of regions.
            A '1' indicates that the corresponding region participates in that subspace.
            Example for N=3:
                '100' -> region 1 specific
                '110' -> shared between regions 1 and 2
                '111' -> shared across all three regions
        num_timeframes (int): Number of time steps in each sequence.
        nhead (int): Number of attention heads in each transformer.
        num_layers (int): Default number of transformer layers per region.
        num_layers_per_region (List[int], optional): Number of encoder/decoder layers for each region.
        max_len (int): Maximum supported sequence length.
        pe (bool): Whether to use positional encoding.
        pe_learn (bool): If True, use learnable positional encoding; otherwise use sinusoidal encoding.
        batch_norm (bool): Whether to apply batch normalization after the encoder projection.
        segment_order (List[str], optional): Optional explicit ordering of latent subspaces.

    Input:
        x:
            - list of N tensors, where region i has shape (B, T, d_i), or
            - a single concatenated tensor of shape (B, T, sum(d_i))

        decode_shared_keys (List[str], optional):
            List of bitstring subspaces to decode separately into `recons_shared`.
            If None, defaults to the all-shared subspace if present.

    Returns:
        recons_self:
            List of N tensors, where region i reconstruction has shape (B, T, d_i).

        recons_shared:
            Dictionary mapping each requested shared key to a list of N decoded tensors.
            For a key `k`, `recons_shared[k][i]` has shape (B, T, d_i) and represents
            reconstruction of region i using only latent subspace `k`.

        latents:
            Dictionary containing:
                - "z_all":
                    Tensor of shape (T, B, D).
                    Fused latent representation obtained by masked averaging across regions.
                - "z_proj":
                    List of N tensors, each of shape (T, B, D).
                    Region-specific latent representations after encoder and linear projection.
                - "region_masks":
                    Tensor of shape (N, D).
                    Binary mask indicating which latent dimensions are used by each region.
                - "segments":
                    List of tuples (key, (start, end)) describing how the full latent
                    vector is partitioned into subspaces.
                - "subset_keys":
                    List of bitstring keys corresponding to the latent subspaces.
                - "latent_gate":
                    Tensor of shape (D,).
                    Multiplicative gate applied to latent dimensions during decoding.
    """

    def __init__(
        self,
        input_dim: List[int],
        dimension: Dict[str, int],
        num_timeframes: int,
        nhead: int = 2,
        num_layers: int = 2,
        num_layers_per_region: List[int] = None,
        max_len: int = 5000,
        pe: bool = False,
        pe_learn: bool = False,
        batch_norm: bool = True,
        segment_order: List[str] = None
    ):
        super().__init__()
        assert len(input_dim) >= 2, "Use N>=2"
        self.N = len(input_dim)
        for k in dimension.keys():
            assert len(k) == self.N, f"Bitstring key '{k}' must have length N={self.N}"

        segments, D, region_masks_1d, subset_masks_1d = build_latent_layout_and_masks(
            dimension, self.N, segment_order=segment_order
        )
        self.segments = segments
        self.latent_dim = D
        self.dimension = dimension
        self.subset_masks_keys = list(subset_masks_1d.keys())

        self.register_buffer("_region_masks_1d", torch.stack(region_masks_1d, dim=0), persistent=False)
        self._subset_key_to_idx = {k: i for i, k in enumerate(self.subset_masks_keys)}
        self.register_buffer(
            "_subset_masks_1d",
            torch.stack([subset_masks_1d[k] for k in self.subset_masks_keys], dim=0),
            persistent=False
        )

        if num_layers_per_region is None:
            num_layers_per_region = [num_layers] * self.N

        self.encoders = nn.ModuleList([
            TransformerEncoder(input_dim[i], self.latent_dim, nhead, num_layers_per_region[i],
                            max_len, pe, pe_learn, batch_norm=batch_norm)
            for i in range(self.N)
        ])
        self.decoders = nn.ModuleList([
            TransformerDecoder(self.latent_dim, input_dim[i], nhead, num_layers_per_region[i],
                            max_len, pe, pe_learn)
            for i in range(self.N)
        ])
        self.linear_proj = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim, bias=False) for _ in range(self.N)
        ])

        self.input_dim = input_dim
        self.cum_splits = [0]
        for d in input_dim:
            self.cum_splits.append(self.cum_splits[-1] + d)

        self.all_shared_key = '1' * self.N if ('1' * self.N) in self.dimension else None

        self.register_buffer("latent_gate", torch.ones(self.latent_dim))

    def _split_concat(self, x_cat: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i in range(self.N):
            a = self.cum_splits[i]; b = self.cum_splits[i+1]
            outs.append(x_cat[..., a:b])
        return outs

    def _ensure_region_list(self, x) -> List[torch.Tensor]:
        if isinstance(x, (list, tuple)):
            assert len(x) == self.N, f"Expected {self.N} region tensors, got {len(x)}"
            return list(x)
        assert x.shape[-1] == self.cum_splits[-1], \
            f"Last dim {x.shape[-1]} != sum(input_dim) {self.cum_splits[-1]}"
        return self._split_concat(x)

    def forward(self, x, decode_shared_keys: List[str] = None):
        """
        Encode each region, fuse latent representations, and decode self and shared reconstructions.

        Args:
            x: Multi-region input as a list of region tensors or a concatenated tensor.
            decode_shared_keys: Latent subspaces to decode separately.

        Returns:
            recons_self, recons_shared, latents
        """
        xs = self._ensure_region_list(x)  # [(B,T,d_i)...]
        B, T = xs[0].shape[:2]
        xs_tb = [xi.permute(1, 0, 2) for xi in xs]  # -> (T,B,d_i)

        z_enc  = [self.encoders[i](xs_tb[i])  for i in range(self.N)]   # (T,B,D)
        z_proj = [self.linear_proj[i](z_enc[i]) for i in range(self.N)] # (T,B,D)

        device = z_proj[0].device
        region_masks = self._region_masks_1d.to(device)                  # (N,D)
        region_masks_b = [region_masks[i].view(1,1,-1) for i in range(self.N)]


        gate_b = self.latent_gate.view(1, 1, -1)                         # (1,1,D)

        
        num = 0; den = 0
        for i in range(self.N):
            wi = region_masks_b[i]
            num = num + (z_proj[i] * wi)
            den = den + wi
        z_all = num / den

        # per-region decode memory = z_all * (w_r * gate)
        dec_inputs = [z_all * (region_masks_b[i] * gate_b) for i in range(self.N)]

        tgt_list = [torch.zeros_like(dec_inputs[i]) for i in range(self.N)]
        recons_self_tb = [self.decoders[i](tgt_list[i], dec_inputs[i]) for i in range(self.N)]
        recons_self = [ri.permute(1, 0, 2) for ri in recons_self_tb]     # -> (B,T,d_i)

        # shared-subspace decodes 
        recons_shared = {}
        if decode_shared_keys is None:
            decode_shared_keys = [self.all_shared_key] if self.all_shared_key is not None else []
        if len(decode_shared_keys) > 0:
            subset_masks_1d = self._subset_masks_1d.to(device)           # (S,D)
            for key in decode_shared_keys:
                if key is None: continue
                assert key in self._subset_key_to_idx, f"Requested shared key '{key}' not in dimension dict."
                kidx = self._subset_key_to_idx[key]
                mk = subset_masks_1d[kidx].view(1, 1, -1)                # (1,1,D)
                z_k = z_all * (mk * gate_b)
                outs_k_tb = [self.decoders[i](torch.zeros_like(z_k), z_k) for i in range(self.N)]
                recons_shared[key] = [ok.permute(1, 0, 2) for ok in outs_k_tb]

        latents = {
            "z_all": z_all,                 # (T,B,D)
            "z_proj": z_proj,               # list of (T,B,D)
            "region_masks": region_masks,   # (N,D)
            "segments": self.segments,
            "subset_keys": self.subset_masks_keys,
            "latent_gate": self.latent_gate
        }
        return recons_self, recons_shared, latents

