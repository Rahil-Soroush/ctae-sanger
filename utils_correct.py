import numpy as np
import os
import json
import torch
import torch.linalg as LA
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold

def create_train_val_loaders(
    data,
    batch_size=8,
    train_ratio=0.8,
    shuffle=True,
    seed=0,
):
    """
    Split an already-predefined training tensor into train/validation loaders.

    Args:
        data: np.ndarray or torch.Tensor with shape [N, T, C]
        batch_size: batch size
        train_ratio: fraction used for training; rest used for validation
        shuffle: whether to shuffle training batches
        seed: random seed for reproducible split

    Returns:
        train_loader, val_loader
    """

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = data.float()

    dataset = TensorDataset(data)

    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def create_test_loader(
    data,
    batch_size=8,
):
    """
    Create a test loader from a separate held-out test tensor.

    Args:
        data: np.ndarray or torch.Tensor with shape [N, T, C]
        batch_size: batch size

    Returns:
        test_loader
    """

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = data.float()

    dataset = TensorDataset(data)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader

def create_data_loaders(
    data,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    y=None,
    y_pos=None,
    seed=0,
):
    """
    Split data into train/validation/test sets and return PyTorch DataLoaders.

    Args:
        data (np.ndarray or torch.Tensor): Input data of shape (N, T, D) or similar.
        batch_size (int): Batch size for DataLoaders.
        train_ratio (float): Fraction of data used for training.
        val_ratio (float): Fraction of data used for validation.
        test_ratio (float): Fraction of data used for testing.
        shuffle (bool): Whether to shuffle training batches.
        y (np.ndarray or torch.Tensor, optional): Labels or targets aligned with data.
        y_pos (np.ndarray or torch.Tensor, optional): Additional targets, e.g., position.
        seed (int): Random seed for reproducible splits.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8, (
        "Train, validation, and test ratios must sum to 1"
    )

    if isinstance(data, np.ndarray):
        data_cur = torch.tensor(data, dtype=torch.float32)
    else:
        data_cur = data.float() if torch.is_tensor(data) else torch.tensor(data, dtype=torch.float32)

    tensors = [data_cur]

    if y is not None:
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        tensors.append(y)

    if y_pos is not None:
        if isinstance(y_pos, np.ndarray):
            y_pos = torch.tensor(y_pos, dtype=torch.float32)
        tensors.append(y_pos)

    dataset = TensorDataset(*tensors)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def my_r2_score(true, pred):
    """
    Compute R² score from predictions.
    """
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    rss = mean_squared_error(true_flat, pred_flat) * len(true_flat)
    tss = ((true_flat - true_flat.mean()) ** 2).sum()
    if tss == 0:
        return np.nan
    return 1 - rss / tss


def compute_ortho_loss(z_shared, z_specific):
    """
    Compute orthogonality loss between shared and specific latent subspaces.

    Args:
        z_shared (torch.Tensor): Shared latent representations (B, T, D_shared).
        z_specific (torch.Tensor): Specific latent representations (B, T, D_specific).

    Returns:
        torch.Tensor: Scalar orthogonality loss.
    """
    batch_size = z_shared.shape[0]
    time_frames = z_shared.shape[1]
    shared_dim = z_shared.shape[2]
    specific_dim = z_specific.shape[2]

    dot_product = torch.bmm(
        z_shared.permute(1, 2, 0),
        z_specific.permute(1, 0, 2),
    )

    orthogonality_loss = torch.norm(dot_product, p="fro") ** 2
    normalization_factor = batch_size * time_frames * shared_dim * specific_dim
    normalized_orthogonality_loss = orthogonality_loss / normalization_factor
    return normalized_orthogonality_loss


def safe_format(v):
    """
    Format values, especially floats, for consistent string representation.
    """
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def pred_normalize(x):
    """
    Normalize prediction targets across trials and time.
    """
    x_flat = x.reshape(-1, x.shape[-1])
    mean = x_flat.mean(axis=0)
    std = x_flat.std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x - mean) / std
    return x_norm


def get_cv_score_acc(X, y):
    """
    Perform cross-validated classification and return accuracy scores.
    """
    random_seed = 0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    clf_template = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        random_state=random_seed,
    )

    y_pred = np.empty_like(y)
    fold_scores = []

    for train_idx, test_idx in cv.split(X, y):
        clf = clone(clf_template)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
        fold_scores.append(accuracy_score(y[test_idx], y_pred[test_idx]))

    fold_scores = np.array(fold_scores)
    return fold_scores, y_pred


def _unpack_batch(batch):
    """
    Handles TensorDataset batches with either only data or data plus labels.
    The original CTAE code assumed every batch was (data_cur, condition).
    """
    if isinstance(batch, (list, tuple)):
        data_cur = batch[0]
        condition = batch[1] if len(batch) > 1 else None
    else:
        data_cur = batch
        condition = None
    return data_cur, condition


def _lambda_ortho_for_epoch(epoch, lambda_ortho, warm_up_ortho):
    if warm_up_ortho <= 0:
        return lambda_ortho
    if epoch < warm_up_ortho:
        return 0.0
    if epoch < 2 * warm_up_ortho:
        progress = (epoch - warm_up_ortho + 1) / warm_up_ortho
        return lambda_ortho * progress
    return lambda_ortho


def _compute_corr_orthogonality(shared_subspace1, shared_subspace2, specific_subspace1, specific_subspace2):
    shared_subspace = (shared_subspace1 + shared_subspace2) / 2
    z = torch.cat([shared_subspace, specific_subspace1, specific_subspace2], dim=2)

    z_flat = z.reshape(-1, z.shape[-1])
    # Guard against numerical issues if a dimension is nearly constant.
    z_flat = z_flat - z_flat.mean(dim=0, keepdim=True)
    z_std = z_flat.std(dim=0, keepdim=True).clamp_min(1e-8)
    z_flat = z_flat / z_std

    corr_matrix = torch.corrcoef(z_flat.T)
    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.eye(corr_matrix.shape[0], device=corr_matrix.device)
    orthogonality_loss = LA.matrix_norm(corr_matrix * (1 - mask))
    return orthogonality_loss


def train_ctae(
    model,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    num_neurons1=None,
    model_path="best_model.pth",
    lambda_alignment=0.01,
    lambda_ortho=0.01,
    warm_up_ortho=0,
    lambda_recons2=1,
    print_every=10,
):
    """
    Train Coupled Transformer Autoencoder (CTAE) model.

    Notes:
        Optimizes reconstruction, cross-reconstruction, shared alignment,
        and orthogonality losses. Best model is selected by validation R².
    """
    best_val_r2score = -float("inf")

    lambda_recons1 = 1.0
    lambda_recons12 = lambda_recons2
    lambda_recons21 = 1.0

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss12 = 0.0
        train_loss21 = 0.0
        train_alignment_loss = 0.0
        train_ortho_loss = 0.0
        train_r2Scores_dataset1 = []
        train_r2Scores_dataset2 = []
        train_r2Scores_dataset12 = []
        train_r2Scores_dataset21 = []

        model.train()
        for batch in train_loader:
            data_cur, _ = _unpack_batch(batch)
            data_cur = data_cur.float().to(device)

            (
                x11_hat,
                x22_hat,
                x12_hat,
                x21_hat,
                shared_subspace1,
                shared_subspace2,
                specific_subspace1,
                specific_subspace2,
                z,
            ) = model(data_cur, num_neurons1=num_neurons1)

            loss1 = criterion(x11_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1]) / num_neurons1
            loss2 = criterion(x22_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:]) / (
                data_cur.shape[-1] - num_neurons1
            )
            loss12 = criterion(x12_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:]) / (
                data_cur.shape[-1] - num_neurons1
            )
            loss21 = criterion(x21_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1]) / num_neurons1

            alignment_loss = criterion(shared_subspace1, shared_subspace2) / shared_subspace1.shape[-1]
            orthogonality_loss = _compute_corr_orthogonality(
                shared_subspace1,
                shared_subspace2,
                specific_subspace1,
                specific_subspace2,
            )

            x1, x2 = model.split_data(data_cur, num_neurons1=num_neurons1)
            data_cur1 = x1.permute(1, 0, 2).detach().cpu().numpy()
            data_cur2 = x2.permute(1, 0, 2).detach().cpu().numpy()
            outputs1 = x11_hat.permute(1, 0, 2).detach().cpu().numpy()
            outputs2 = x22_hat.permute(1, 0, 2).detach().cpu().numpy()
            outputs21 = x21_hat.permute(1, 0, 2).detach().cpu().numpy()
            outputs12 = x12_hat.permute(1, 0, 2).detach().cpu().numpy()

            train_r2Scores_dataset1.extend(
                [my_r2_score(data_cur1[:, :, i], outputs1[:, :, i]) for i in range(outputs1.shape[-1])]
            )
            train_r2Scores_dataset2.extend(
                [my_r2_score(data_cur2[:, :, i], outputs2[:, :, i]) for i in range(outputs2.shape[-1])]
            )
            train_r2Scores_dataset21.extend(
                [my_r2_score(data_cur1[:, :, i], outputs21[:, :, i]) for i in range(outputs1.shape[-1])]
            )
            train_r2Scores_dataset12.extend(
                [my_r2_score(data_cur2[:, :, i], outputs12[:, :, i]) for i in range(outputs2.shape[-1])]
            )

            lambda_ortho_epoch = _lambda_ortho_for_epoch(epoch, lambda_ortho, warm_up_ortho)

            loss = (
                lambda_recons1 * loss1
                + lambda_recons2 * loss2
                + lambda_recons12 * loss12
                + lambda_recons21 * loss21
                + lambda_alignment * alignment_loss
                + lambda_ortho_epoch * orthogonality_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data_cur.size(0)
            train_loss1 += loss1.item() * data_cur.size(0)
            train_loss2 += loss2.item() * data_cur.size(0)
            train_loss12 += loss12.item() * data_cur.size(0)
            train_loss21 += loss21.item() * data_cur.size(0)
            train_alignment_loss += alignment_loss.item() * data_cur.size(0)
            train_ortho_loss += orthogonality_loss.item() * data_cur.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss1 /= len(train_loader.dataset)
        train_loss2 /= len(train_loader.dataset)
        train_loss12 /= len(train_loader.dataset)
        train_loss21 /= len(train_loader.dataset)
        train_alignment_loss /= len(train_loader.dataset)
        train_ortho_loss /= len(train_loader.dataset)
        train_r2score1 = np.nanmean(train_r2Scores_dataset1)
        train_r2score2 = np.nanmean(train_r2Scores_dataset2)
        train_r2score12 = np.nanmean(train_r2Scores_dataset12)
        train_r2score21 = np.nanmean(train_r2Scores_dataset21)
        train_r2score = (train_r2score1 + train_r2score2) / 2

        if epoch % print_every != 0:
            continue

        val_loss = 0.0
        val_loss1 = 0.0
        val_loss2 = 0.0
        val_loss12 = 0.0
        val_loss21 = 0.0
        val_alignment_loss = 0.0
        val_ortho_loss = 0.0
        val_r2Scores_dataset1 = []
        val_r2Scores_dataset2 = []
        val_r2Scores_dataset12 = []
        val_r2Scores_dataset21 = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                data_cur, _ = _unpack_batch(batch)
                data_cur = data_cur.float().to(device)

                (
                    x11_hat,
                    x22_hat,
                    x12_hat,
                    x21_hat,
                    shared_subspace1,
                    shared_subspace2,
                    specific_subspace1,
                    specific_subspace2,
                    z,
                ) = model(data_cur, num_neurons1=num_neurons1)

                loss1 = criterion(x11_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1]) / num_neurons1
                loss2 = criterion(x22_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:]) / (
                    data_cur.shape[-1] - num_neurons1
                )
                loss12 = criterion(x12_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:]) / (
                    data_cur.shape[-1] - num_neurons1
                )
                loss21 = criterion(x21_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1]) / num_neurons1

                alignment_loss = criterion(shared_subspace1, shared_subspace2) / shared_subspace1.shape[-1]
                orthogonality_loss = _compute_corr_orthogonality(
                    shared_subspace1,
                    shared_subspace2,
                    specific_subspace1,
                    specific_subspace2,
                )

                x1, x2 = model.split_data(data_cur, num_neurons1=num_neurons1)
                data_cur1 = x1.permute(1, 0, 2).detach().cpu().numpy()
                data_cur2 = x2.permute(1, 0, 2).detach().cpu().numpy()
                outputs1 = x11_hat.permute(1, 0, 2).detach().cpu().numpy()
                outputs2 = x22_hat.permute(1, 0, 2).detach().cpu().numpy()
                outputs21 = x21_hat.permute(1, 0, 2).detach().cpu().numpy()
                outputs12 = x12_hat.permute(1, 0, 2).detach().cpu().numpy()

                val_r2Scores_dataset1.extend(
                    [my_r2_score(data_cur1[:, :, i], outputs1[:, :, i]) for i in range(outputs1.shape[-1])]
                )
                val_r2Scores_dataset2.extend(
                    [my_r2_score(data_cur2[:, :, i], outputs2[:, :, i]) for i in range(outputs2.shape[-1])]
                )
                val_r2Scores_dataset21.extend(
                    [my_r2_score(data_cur1[:, :, i], outputs21[:, :, i]) for i in range(outputs1.shape[-1])]
                )
                val_r2Scores_dataset12.extend(
                    [my_r2_score(data_cur2[:, :, i], outputs12[:, :, i]) for i in range(outputs2.shape[-1])]
                )

                lambda_ortho_epoch = _lambda_ortho_for_epoch(epoch, lambda_ortho, warm_up_ortho)
                loss = (
                    lambda_recons1 * loss1
                    + lambda_recons2 * loss2
                    + lambda_recons12 * loss12
                    + lambda_recons21 * loss21
                    + lambda_alignment * alignment_loss
                    + lambda_ortho_epoch * orthogonality_loss
                )

                val_loss += loss.item() * data_cur.size(0)
                val_loss1 += loss1.item() * data_cur.size(0)
                val_loss2 += loss2.item() * data_cur.size(0)
                val_loss12 += loss12.item() * data_cur.size(0)
                val_loss21 += loss21.item() * data_cur.size(0)
                val_alignment_loss += alignment_loss.item() * data_cur.size(0)
                val_ortho_loss += orthogonality_loss.item() * data_cur.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss1 /= len(val_loader.dataset)
        val_loss2 /= len(val_loader.dataset)
        val_loss12 /= len(val_loader.dataset)
        val_loss21 /= len(val_loader.dataset)
        val_alignment_loss /= len(val_loader.dataset)
        val_ortho_loss /= len(val_loader.dataset)
        val_r2score1 = np.nanmean(val_r2Scores_dataset1)
        val_r2score2 = np.nanmean(val_r2Scores_dataset2)
        val_r2score12 = np.nanmean(val_r2Scores_dataset12)
        val_r2score21 = np.nanmean(val_r2Scores_dataset21)
        val_r2score = (val_r2score1 + val_r2score2) / 2

        print(
            f"Epoch {epoch:04d} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"train R2 {train_r2score:.3f} | val R2 {val_r2score:.3f} | "
            f"val R2 x1 {val_r2score1:.3f}, x2 {val_r2score2:.3f}, "
            f"x12 {val_r2score12:.3f}, x21 {val_r2score21:.3f}"
        )

        if val_r2score > best_val_r2score:
            best_val_r2score = val_r2score
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with validation R2: {best_val_r2score:.3f} at epoch {epoch}")



def train_ctae_with_logging(
    model,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    num_neurons1,
    model_path,
    lambda_alignment=0.01,
    lambda_ortho=0.01,
    warm_up_ortho=0,
    lambda_recons2=1.0,
    early_stopping=True,
    patience=20,
    min_delta=1e-4,
    start_epoch=50,
    log_dir=None,
    bundle_path=None,
    metadata=None,
    save_every_best=True,
):
    """
    CTAE training function with TensorBoard logging and reloadable bundle saving.

    Saves:
        model_path: best model state_dict
        bundle_path: dict containing model, state_dict, optimizer, history, metadata
    """

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if bundle_path is None:
        bundle_path = model_path.replace(".pth", "_bundle.pth").replace(".pt", "_bundle.pth")

    if log_dir is None:
        log_dir = model_path.replace(".pth", "_tb").replace(".pt", "_tb")

    writer = SummaryWriter(log_dir=log_dir)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_loss1": [],
        "train_loss2": [],
        "train_loss12": [],
        "train_loss21": [],
        "train_alignment_loss": [],
        "train_ortho_loss": [],
        "train_r2_own_mean": [],
        "train_r2_gpi": [],
        "train_r2_stn": [],
        "train_r2_cross_gpi_from_stn": [],
        "train_r2_cross_stn_from_gpi": [],
        "val_loss": [],
        "val_loss1": [],
        "val_loss2": [],
        "val_loss12": [],
        "val_loss21": [],
        "val_alignment_loss": [],
        "val_ortho_loss": [],
        "val_r2_own_mean": [],
        "val_r2_gpi": [],
        "val_r2_stn": [],
        "val_r2_cross_gpi_from_stn": [],
        "val_r2_cross_stn_from_gpi": [],
        "lambda_ortho_epoch": [],
    }

    best_val_r2score = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1

    lambda_recons1 = 1.0
    lambda_recons12 = lambda_recons2
    lambda_recons21 = 1.0

    def get_lambda_ortho_epoch(epoch):
        if warm_up_ortho <= 0:
            return lambda_ortho
        if epoch < warm_up_ortho:
            return 0.0
        elif epoch < 2 * warm_up_ortho:
            progress = (epoch - warm_up_ortho + 1) / warm_up_ortho
            return lambda_ortho * progress
        else:
            return lambda_ortho

    def r2_score_np(true, pred, eps=1e-12):
        true = true.reshape(-1)
        pred = pred.reshape(-1)
        rss = np.sum((true - pred) ** 2)
        tss = np.sum((true - true.mean()) ** 2)
        return 1.0 - rss / (tss + eps)

    def run_epoch(loader, train=True, epoch=0):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss12 = 0.0
        total_loss21 = 0.0
        total_alignment = 0.0
        total_ortho = 0.0

        r2_1_all = []
        r2_2_all = []
        r2_12_all = []
        r2_21_all = []

        lambda_ortho_epoch = get_lambda_ortho_epoch(epoch)

        for batch in loader:
            data_cur, _ = _unpack_batch(batch)
            data_cur = data_cur.float().to(device)

            with torch.set_grad_enabled(train):
                (
                    x11_hat,
                    x22_hat,
                    x12_hat,
                    x21_hat,
                    shared_subspace1,
                    shared_subspace2,
                    specific_subspace1,
                    specific_subspace2,
                    z,
                ) = model(data_cur, num_neurons1=num_neurons1)

                loss1 = criterion(
                    x11_hat.permute(1, 0, 2),
                    data_cur[:, :, :num_neurons1],
                ) / num_neurons1

                loss2 = criterion(
                    x22_hat.permute(1, 0, 2),
                    data_cur[:, :, num_neurons1:],
                ) / (data_cur.shape[-1] - num_neurons1)

                loss12 = criterion(
                    x12_hat.permute(1, 0, 2),
                    data_cur[:, :, num_neurons1:],
                ) / (data_cur.shape[-1] - num_neurons1)

                loss21 = criterion(
                    x21_hat.permute(1, 0, 2),
                    data_cur[:, :, :num_neurons1],
                ) / num_neurons1

                alignment_loss = criterion(
                    shared_subspace1,
                    shared_subspace2,
                ) / shared_subspace1.shape[-1]

                shared_subspace = (shared_subspace1 + shared_subspace2) / 2
                z_all = torch.cat(
                    [shared_subspace, specific_subspace1, specific_subspace2],
                    dim=2,
                )

                z_flat = z_all.reshape(-1, z_all.shape[-1])

                if z_flat.shape[0] > 1:
                    corr_matrix = torch.corrcoef(z_flat.T)
                    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                    mask = torch.eye(
                        corr_matrix.shape[0],
                        corr_matrix.shape[1],
                        device=corr_matrix.device,
                    )
                    orthogonality_loss = LA.matrix_norm(corr_matrix * (1 - mask))
                else:
                    orthogonality_loss = torch.tensor(0.0, device=device)

                loss = (
                    lambda_recons1 * loss1
                    + lambda_recons2 * loss2
                    + lambda_recons12 * loss12
                    + lambda_recons21 * loss21
                    + lambda_alignment * alignment_loss
                    + lambda_ortho_epoch * orthogonality_loss
                )

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            bs = data_cur.size(0)
            total_loss += loss.item() * bs
            total_loss1 += loss1.item() * bs
            total_loss2 += loss2.item() * bs
            total_loss12 += loss12.item() * bs
            total_loss21 += loss21.item() * bs
            total_alignment += alignment_loss.item() * bs
            total_ortho += orthogonality_loss.item() * bs

            x1, x2 = model.split_data(data_cur, num_neurons1=num_neurons1)

            true1 = x1.permute(1, 0, 2).detach().cpu().numpy()
            true2 = x2.permute(1, 0, 2).detach().cpu().numpy()
            pred1 = x11_hat.permute(1, 0, 2).detach().cpu().numpy()
            pred2 = x22_hat.permute(1, 0, 2).detach().cpu().numpy()
            pred21 = x21_hat.permute(1, 0, 2).detach().cpu().numpy()
            pred12 = x12_hat.permute(1, 0, 2).detach().cpu().numpy()

            for ch in range(pred1.shape[-1]):
                r2_1_all.append(r2_score_np(true1[:, :, ch], pred1[:, :, ch]))
                r2_21_all.append(r2_score_np(true1[:, :, ch], pred21[:, :, ch]))

            for ch in range(pred2.shape[-1]):
                r2_2_all.append(r2_score_np(true2[:, :, ch], pred2[:, :, ch]))
                r2_12_all.append(r2_score_np(true2[:, :, ch], pred12[:, :, ch]))

        n = len(loader.dataset)

        metrics = {
            "loss": total_loss / n,
            "loss1": total_loss1 / n,
            "loss2": total_loss2 / n,
            "loss12": total_loss12 / n,
            "loss21": total_loss21 / n,
            "alignment_loss": total_alignment / n,
            "ortho_loss": total_ortho / n,
            "r2_gpi": float(np.mean(r2_1_all)),
            "r2_stn": float(np.mean(r2_2_all)),
            "r2_cross_gpi_from_stn": float(np.mean(r2_21_all)),
            "r2_cross_stn_from_gpi": float(np.mean(r2_12_all)),
            "lambda_ortho_epoch": lambda_ortho_epoch,
        }

        metrics["r2_own_mean"] = 0.5 * (metrics["r2_gpi"] + metrics["r2_stn"])

        return metrics
    
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        train_metrics = run_epoch(train_loader, train=True, epoch=epoch)
        val_metrics = run_epoch(val_loader, train=False, epoch=epoch)

        # TensorBoard logging
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        writer.add_scalar("hyperparams/lambda_alignment", lambda_alignment, epoch)
        writer.add_scalar("hyperparams/lambda_ortho_epoch", train_metrics["lambda_ortho_epoch"], epoch)

        # Store history
        history["epoch"].append(epoch)

        for k in [
            "loss", "loss1", "loss2", "loss12", "loss21",
            "alignment_loss", "ortho_loss", "r2_own_mean",
            "r2_gpi", "r2_stn", "r2_cross_gpi_from_stn",
            "r2_cross_stn_from_gpi"
        ]:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])

        history["lambda_ortho_epoch"].append(train_metrics["lambda_ortho_epoch"])

        # Save best by validation own-region R2
        current_val_r2 = val_metrics["r2_own_mean"]
        current_val_loss = val_metrics["loss"]

        # improved = current_val_r2 > best_val_r2score
        improved = current_val_r2 > best_val_r2score + min_delta

        if improved:
            best_val_r2score = current_val_r2
            best_val_loss = current_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_path)

            bundle = {
                "model": model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "metadata": metadata,
                "best_epoch": best_epoch,
                "best_val_r2score": best_val_r2score,
                "best_val_loss": best_val_loss,
                "num_neurons1": num_neurons1,
                "lambda_alignment": lambda_alignment,
                "lambda_ortho": lambda_ortho,
                "warm_up_ortho": warm_up_ortho,
                "lambda_recons2": lambda_recons2,
            }

            torch.save(bundle, bundle_path)

            print(
                f"Saved best model | epoch {epoch} | "
                f"val R2={best_val_r2score:.4f} | val loss={best_val_loss:.6f}"
            )
        else:
            if early_stopping and epoch >= start_epoch:
                epochs_without_improvement += 1
                
        if early_stopping and epoch >= start_epoch and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best val R2: {best_val_r2score:.4f}"
            )
            break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train loss={train_metrics['loss']:.6f}, "
                f"val loss={val_metrics['loss']:.6f}, "
                f"train R2={train_metrics['r2_own_mean']:.4f}, "
                f"val R2={val_metrics['r2_own_mean']:.4f}, "
                f"align={val_metrics['alignment_loss']:.6f}, "
                f"ortho={val_metrics['ortho_loss']:.6f}"
            )

    writer.close()

    print(f"Training complete. Best epoch: {best_epoch}")
    print(f"Best model state_dict saved to: {model_path}")
    print(f"Best bundle saved to: {bundle_path}")
    print(f"TensorBoard logs saved to: {log_dir}")

    return history, model