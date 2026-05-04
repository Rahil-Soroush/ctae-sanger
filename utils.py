import numpy as np

import torch
import torch.linalg as LA
from torch.utils.data import DataLoader, random_split, TensorDataset

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold


def create_data_loaders(
    data,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    y=None,
    y_pos=None,
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
        y_pos (np.ndarray or torch.Tensor, optional): Additional targets (e.g., position).

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.

    """

    assert train_ratio + val_ratio + test_ratio == 1, "Train, validation, and test ratios must sum to 1"

    if isinstance(data, np.ndarray):
        data_cur = torch.tensor(data, dtype=torch.float32)
    else:
        data_cur = data

    tensors = [data_cur]

    if y is not None:
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        tensors.append(y)

    if y_pos is not None:
        if isinstance(y_pos, np.ndarray):
            y_pos = torch.tensor(y_pos, dtype=torch.float32)
        tensors.append(y_pos)

    data_cur = TensorDataset(*tensors)

    dataset_size = len(data_cur)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        data_cur, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def my_r2_score(true,pred):
    """
    Compute R² score from predictions.

    Args:
        true (np.ndarray): Ground truth values.
        pred (np.ndarray): Predicted values.

    Returns:
        float: R² score.
    """
    rss = mean_squared_error(true.flatten(), pred.flatten()) * len(true.flatten())
    tss = ((true.flatten() - true.flatten().mean()) ** 2).sum()
    return 1 - rss / tss

def compute_ortho_loss(z_shared,z_specific):
    """
    Compute orthogonality loss between shared and specific latent subspaces.

    Args:
        z_shared (torch.Tensor): Shared latent representations (B, T, D_shared).
        z_specific (torch.Tensor): Specific latent representations (B, T, D_specific).

    Returns:
        torch.Tensor: Scalar orthogonality loss.
    """
    batch_size = z_shared.shape[0]        # Batch size
    time_frames = z_shared.shape[1]       # Time frames
    shared_dim = z_shared.shape[2]        # Shared latent dimension
    specific_dim = z_specific.shape[2]    # Specific latent dimension

    # Compute dot product
    dot_product = torch.bmm(z_shared.permute(1, 2, 0), z_specific.permute(1,0,2)) # Shape: (batch_size, shared_dim, specific_dim)

    # Compute Frobenius norm of dot product
    orthogonality_loss = torch.norm(dot_product, p='fro') ** 2

    # Normalize the loss
    normalization_factor = batch_size * time_frames * shared_dim * specific_dim
    normalized_orthogonality_loss = orthogonality_loss / normalization_factor
    return normalized_orthogonality_loss


def safe_format(v):
    """
    Format values (especially floats) for consistent string representation.

    Args:
        v (Any): Value to format.

    Returns:
        str: Formatted string.

    Notes:
        - Floats are formatted to 5 significant digits.
        - Used for clean naming/logging.
    """
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def pred_normalize(x):
    """
    Normalize prediction targets across trials and time.

    Args:
        x (np.ndarray): Input array of shape (trials, time, dim).

    Returns:
        np.ndarray: Normalized array with zero mean and unit variance.

    Notes:
        - Normalization is computed globally across trials and time.
        - Applied independently per output dimension.
    """
    # flatten across trials & time
    x_flat = x.reshape(-1, 2)

    # compute mean & std per dimension
    mean  = x_flat.mean(axis=0)      # shape (2,)
    std   = x_flat.std(axis=0)       # shape (2,)

    # apply
    x_norm = (x - mean) / std   # broadcasts back to (trials, times, 2)
    
    return x_norm

def get_cv_score_acc(X,y):
    """
    Perform cross-validated classification and return accuracy scores.

    Args:
        X (np.ndarray): Feature matrix of shape (N, D).
        y (np.ndarray): Labels of shape (N,).

    Returns:
        fold_scores (np.ndarray): Accuracy for each CV fold.
        y_pred (np.ndarray): Cross-validated predictions.
    """
    
    RANDOM_SEED = 0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf_template = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=500,
        random_state=RANDOM_SEED
    )
    
    y_pred = np.empty_like(y)
    fold_scores = []
    
    for train_idx, test_idx in cv.split(X,y):
        clf = clone(clf_template)
        clf.fit(X[train_idx], y[train_idx])

        y_pred[test_idx] = clf.predict(X[test_idx])
        fold_scores.append(accuracy_score(y[test_idx], y_pred[test_idx]))

    fold_scores = np.array(fold_scores)

    return fold_scores,y_pred


def train_ctae(model, train_loader, val_loader, 
               num_epochs, criterion, optimizer, device, 
               num_neurons1=None, model_path="best_model.pth", 
               lambda_alignment=0.01,lambda_ortho=0.01,warm_up_ortho=0,lambda_recons2=1):
    """
    Train Coupled Transformer Autoencoder (CTAE) model.

    Args:
        model (nn.Module): CTAE model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of training epochs.
        criterion (callable): Loss function (e.g., MSE).
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device for computation.
        num_neurons1 (int): Split index for region 1 vs region 2.
        model_path (str): Path to save best model.
        lambda_alignment (float): Weight for shared alignment loss.
        lambda_ortho (float): Weight for orthogonality loss.
        warm_up_ortho (int): Epochs before applying orthogonality loss.
        lambda_recons2 (float): Weight for reconstruction of region 2.

    Returns:
        None

    Notes:
        - Optimizes:
            - Reconstruction losses 
            - Shared only reconstruction loss
            - Alignment loss (shared subspaces)
            - Orthogonality loss (disentanglement)
        - Orthogonality loss is gradually introduced via warm-up.
        - Best model is selected based on validation R² score.
        - Computes per-neuron R² metrics for monitoring reconstruction quality.
    """

    best_val_loss = float('inf')  # Initialize the best validation loss with infinity
    best_val_r2score = -float('inf')

    model.train()
    lambda_recons1 = 1
    lambda_recons12 = lambda_recons2 #lambda_recons1 #* 2
    lambda_recons21 = 1
    lambda_alignment = lambda_alignment


    for epoch in range(num_epochs):
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss12 = 0
        train_loss21 = 0
        train_alignment_loss = 0
        train_ortho_loss = 0
        train_norm_penalty = 0
        shared_subspace1_norm = 0
        shared_subspace2_norm = 0
        train_r2Scores_dataset1 = []
        train_r2Scores_dataset2 = []
        train_r2Scores_dataset12 = []
        train_r2Scores_dataset21 = []

        model.train()
        for data_cur, condition in train_loader:
            data_cur = data_cur.float().to(device)

            # Forward pass
            x11_hat, x22_hat, x12_hat, x21_hat, shared_subspace1, shared_subspace2,specific_subspace1,specific_subspace2,z = model(data_cur, num_neurons1=num_neurons1)

            # Reconstruction losses
            loss1 = criterion(x11_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1])/num_neurons1
            loss2 = criterion(x22_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:])/(data_cur.shape[-1]-num_neurons1)
            loss12 = criterion(x12_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:])/(data_cur.shape[-1]-num_neurons1)
            loss21 = criterion(x21_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1])/num_neurons1

            # Alignment loss
            alignment_loss = criterion(shared_subspace1, shared_subspace2)/shared_subspace1.shape[-1]
            
            # Orthogonality loss
            shared_subspace = (shared_subspace1+shared_subspace2)/2
            z = torch.cat([shared_subspace,specific_subspace1,specific_subspace2],axis=2)
            
            corr_matrix = torch.corrcoef(z.view(-1,z.shape[-1]).T)
            mask = torch.eye(corr_matrix.shape[0],corr_matrix.shape[0])
            orthogonality_loss = LA.matrix_norm(corr_matrix*(1-mask).to(corr_matrix.device)) 

            x1, x2 = model.split_data(data_cur, num_neurons1=num_neurons1)

            data_cur1 = x1.permute(1,0,2).detach().cpu().numpy()
            data_cur2 = x2.permute(1,0,2).detach().cpu().numpy()
            outputs1 = x11_hat.permute(1,0,2).detach().cpu().numpy()
            outputs2 = x22_hat.permute(1,0,2).detach().cpu().numpy()
            outputs21 = x21_hat.permute(1,0,2).detach().cpu().numpy()
            outputs12 = x12_hat.permute(1,0,2).detach().cpu().numpy()

            train_r2Scores_dataset1.extend([my_r2_score(data_cur1[:,:,neuron_idx],outputs1[:,:,neuron_idx]) for neuron_idx in range(outputs1.shape[-1])])

            train_r2Scores_dataset2.extend([my_r2_score(data_cur2[:,:,neuron_idx],outputs2[:,:,neuron_idx]) for neuron_idx in range(outputs2.shape[-1])])   

            train_r2Scores_dataset21.extend([my_r2_score(data_cur1[:,:,neuron_idx],outputs21[:,:,neuron_idx]) for neuron_idx in range(outputs1.shape[-1])])

            train_r2Scores_dataset12.extend([my_r2_score(data_cur2[:,:,neuron_idx],outputs12[:,:,neuron_idx]) for neuron_idx in range(outputs2.shape[-1])])  

            if epoch<warm_up_ortho:
                lambda_ortho_epoch = 0
            elif epoch < 2 * warm_up_ortho:
                # ramp‐up period: epoch runs from warm_up_ortho → 2*warm_up_ortho-1
                # we add +1 so that at epoch=warm_up_ortho we start at 1/warm_up_ortho
                progress = (epoch - warm_up_ortho + 1) / warm_up_ortho
                lambda_ortho_epoch = lambda_ortho * progress
            else:
                lambda_ortho_epoch = lambda_ortho

            # Total loss
            loss = (
                lambda_recons1 * loss1 +
                lambda_recons2 * loss2 +
                lambda_recons12 * loss12 +
                lambda_recons21 * loss21 +
                lambda_alignment * alignment_loss +
                lambda_ortho_epoch * orthogonality_loss
            ) 

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses and norms
            train_loss += loss.item() * data_cur.size(0)
            train_loss1 += loss1.item() * data_cur.size(0)
            train_loss2 += loss2.item() * data_cur.size(0)
            train_loss12 += loss12.item() * data_cur.size(0)
            train_loss21 += loss21.item() * data_cur.size(0)
            train_alignment_loss += alignment_loss.item() * data_cur.size(0)
            train_ortho_loss += orthogonality_loss.item() * data_cur.size(0)

        # Normalize losses and norms by dataset size
        train_loss /= len(train_loader.dataset)
        train_loss1 /= len(train_loader.dataset)
        train_loss2 /= len(train_loader.dataset)
        train_loss12 /= len(train_loader.dataset)
        train_loss21 /= len(train_loader.dataset)
        train_alignment_loss /= len(train_loader.dataset)
        train_ortho_loss /= len(train_loader.dataset)
        train_r2score1 = np.mean(train_r2Scores_dataset1)
        train_r2score2 = np.mean(train_r2Scores_dataset2)
        train_r2score12 = np.mean(train_r2Scores_dataset12)
        train_r2score21 = np.mean(train_r2Scores_dataset21)
        train_r2score = (train_r2score1+train_r2score2)/2

        
        # Validation step every few epochs
        if epoch % 10 == 0:
            val_loss = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss12 = 0
            val_loss21 = 0
            val_alignment_loss = 0
            val_ortho_loss = 0
            val_norm_penalty = 0
            val_shared_subspace1_norm = 0
            val_shared_subspace2_norm = 0
            val_r2Scores_dataset1 = []
            val_r2Scores_dataset2 = []
            val_r2Scores_dataset12 = []
            val_r2Scores_dataset21 = []

            model.eval()
            with torch.no_grad():
                for data_cur, condition in val_loader:
                    data_cur = data_cur.float().to(device)


                    x11_hat, x22_hat, x12_hat, x21_hat, shared_subspace1, shared_subspace2, specific_subspace1,specific_subspace2,z = model(data_cur, num_neurons1=num_neurons1)

                    # Reconstruction losses
                    loss1 = criterion(x11_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1])/num_neurons1
                    loss2 = criterion(x22_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:])/(data_cur.shape[-1]-num_neurons1)
                    loss12 = criterion(x12_hat.permute(1, 0, 2), data_cur[:, :, num_neurons1:])/(data_cur.shape[-1]-num_neurons1)
                    loss21 = criterion(x21_hat.permute(1, 0, 2), data_cur[:, :, :num_neurons1])/num_neurons1

                    # Alignment loss
                    alignment_loss = criterion(shared_subspace1, shared_subspace2)/shared_subspace1.shape[-1]

                    shared_subspace = (shared_subspace1+shared_subspace2)/2
                    z = torch.cat([shared_subspace,specific_subspace1,specific_subspace2],axis=2)
                    corr_matrix = torch.corrcoef(z.view(-1,z.shape[-1]).T)
                    mask = torch.eye(corr_matrix.shape[0],corr_matrix.shape[0])

                    tc =  LA.matrix_norm(corr_matrix*(1-mask).to(corr_matrix.device)) 
                    orthogonality_loss = tc

                    x1, x2 = model.split_data(data_cur, num_neurons1=num_neurons1)

                    data_cur1 = x1.permute(1,0,2).detach().cpu().numpy()
                    data_cur2 = x2.permute(1,0,2).detach().cpu().numpy()
                    outputs1 = x11_hat.permute(1,0,2).detach().cpu().numpy()
                    outputs2 = x22_hat.permute(1,0,2).detach().cpu().numpy()
                    outputs21 = x21_hat.permute(1,0,2).detach().cpu().numpy()
                    outputs12 = x12_hat.permute(1,0,2).detach().cpu().numpy()

                    val_r2Scores_dataset1.extend([my_r2_score(data_cur1[:,:,neuron_idx],outputs1[:,:,neuron_idx]) for neuron_idx in range(outputs1.shape[-1])])

                    val_r2Scores_dataset2.extend([my_r2_score(data_cur2[:,:,neuron_idx],outputs2[:,:,neuron_idx]) for neuron_idx in range(outputs2.shape[-1])])   

                    val_r2Scores_dataset21.extend([my_r2_score(data_cur1[:,:,neuron_idx],outputs21[:,:,neuron_idx]) for neuron_idx in range(outputs1.shape[-1])])

                    val_r2Scores_dataset12.extend([my_r2_score(data_cur2[:,:,neuron_idx],outputs12[:,:,neuron_idx]) for neuron_idx in range(outputs2.shape[-1])])   



                    if epoch<warm_up_ortho:
                        lambda_ortho_epoch = 0
                    elif epoch < 2 * warm_up_ortho:
                        # ramp‐up period: epoch runs from warm_up_ortho → 2*warm_up_ortho-1
                        # we add +1 so that at epoch=warm_up_ortho we start at 1/warm_up_ortho
                        progress = (epoch - warm_up_ortho + 1) / warm_up_ortho
                        lambda_ortho_epoch = lambda_ortho * progress
                    else:
                        lambda_ortho_epoch = lambda_ortho

                    # Total loss
                    loss = (
                        lambda_recons1 * loss1 +
                        lambda_recons2 * loss2 +
                        lambda_recons12 * loss12 +
                        lambda_recons21 * loss21 +
                        lambda_alignment * alignment_loss +
                        lambda_ortho_epoch * orthogonality_loss
                    )

                    # Accumulate validation losses and norms
                    val_loss += loss.item() * data_cur.size(0)
                    val_loss1 += loss1.item() * data_cur.size(0)
                    val_loss2 += loss2.item() * data_cur.size(0)
                    val_loss12 += loss12.item() * data_cur.size(0)
                    val_loss21 += loss21.item() * data_cur.size(0)
                    val_alignment_loss += alignment_loss.item() * data_cur.size(0)
                    val_ortho_loss += orthogonality_loss.item() * data_cur.size(0)

            # Normalize validation losses and norms
            val_loss /= len(val_loader.dataset)
            val_loss1 /= len(val_loader.dataset)
            val_loss2 /= len(val_loader.dataset)
            val_loss12 /= len(val_loader.dataset)
            val_loss21 /= len(val_loader.dataset)
            val_alignment_loss /= len(val_loader.dataset)
            val_ortho_loss /= len(val_loader.dataset)
            val_r2score1 = np.mean(val_r2Scores_dataset1)
            val_r2score2 = np.mean(val_r2Scores_dataset2)
            val_r2score12 = np.mean(val_r2Scores_dataset12)
            val_r2score21 = np.mean(val_r2Scores_dataset21)
            val_r2score = (val_r2score1+val_r2score2)/2

            
            if val_r2score > best_val_r2score: #
                best_val_r2score = val_r2score
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model with validation loss: {best_val_r2score:.2f} at epoch {epoch}")



