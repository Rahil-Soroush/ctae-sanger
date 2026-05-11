# profile_ctae.py — CTAE params + inference ms/seq

import os
import time
import random
import numpy as np
import torch

from models.ctae import CoupledTransformerAutoencoderTwoRegions 



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
AMP = (DEVICE == "cuda" and DTYPE == torch.float16)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def zscore_lfp_tensor(x, eps=1e-8):
    # x: [N, T, C]
    mean = x.reshape(-1, x.shape[-1]).mean(dim=0)
    std = x.reshape(-1, x.shape[-1]).std(dim=0)
    return (x - mean) / (std + eps)


@torch.inference_mode()
def measure_ctae_infer_ms(model, data_batch, num_neurons1, iters=50, warmup=10):
    model.eval()

    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=AMP):
            _ = model(data_batch, num_neurons1=num_neurons1)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=AMP):
            _ = model(data_batch, num_neurons1=num_neurons1)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / iters
    ms_per_seq = ms_per_batch / data_batch.shape[0]

    return ms_per_seq


def main():
    # ------------------------
    # Match compute table setup
    # ------------------------
    subj = "s520"
    data_root = r"F:\comp_project\Off_tensor_Data_R"

    batch_size = 8

    # CTAE hyperparameters
    nhead = 1
    num_layers = 2
    pe = True
    pe_learn = False
    max_len = 300

    # choose one representative subject-specific configuration
    shared_latent_dim = 3
    r1_specific_dim = 3
    r2_specific_dim = 3

    seed = 702
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ------------------------
    # Load data: [N, T, C]
    # ------------------------
    subj_dir = os.path.join(data_root, subj)
    gpi = torch.load(os.path.join(subj_dir, "gpi_train_off.pt"), map_location="cpu").float()
    stn = torch.load(os.path.join(subj_dir, "stn_train_off.pt"), map_location="cpu").float()

    # Match CTAE preprocessing
    N = min(gpi.shape[0], stn.shape[0])
    T = min(gpi.shape[1], stn.shape[1])

    gpi = gpi[:N, :T, :]
    stn = stn[:N, :T, :]

    gpi = zscore_lfp_tensor(gpi)
    stn = zscore_lfp_tensor(stn)

    data = torch.cat([gpi, stn], dim=-1).float()  # [N, T, Cgpi+Cstn]

    input_dim1 = gpi.shape[-1]
    input_dim2 = stn.shape[-1]
    num_neurons1 = input_dim1

    B = min(batch_size, data.shape[0])
    data_b = data[:B].to(DEVICE)

    # ------------------------
    # Create model
    # ------------------------
    model = CoupledTransformerAutoencoderTwoRegions(
        input_dim1,
        input_dim2,
        r1_specific_dim,
        r2_specific_dim,
        shared_latent_dim,
        nhead,
        num_layers,
        num_layers,
        max_len,
        pe,
        pe_learn,
    ).to(DEVICE)

    # Optional: load a trained checkpoint if you want timing on trained model.
    # Timing should be almost identical for random vs trained weights.
    # ckpt_path = r"path\to\your\ctae_model.pth"
    # model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    total_params = count_params(model)
    ms_seq = measure_ctae_infer_ms(
        model=model,
        data_batch=data_b,
        num_neurons1=num_neurons1,
        iters=50,
        warmup=10,
    )

    print(f"[CTAE] Params (M): {total_params / 1e6:.3f}")
    print(
        f"[CTAE] Inference (ms/sequence) @ "
        f"B={B}, T={T}, Cg/Cs={input_dim1}/{input_dim2}, dtype={DTYPE}: {ms_seq:.2f}"
    )


if __name__ == "__main__":
    main()