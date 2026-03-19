import torch
import numpy as np



def verify_globally_disjoint(encodings, R, delta):
    """
    Verifies if a dataset is globally disjoint after rotation.
    """
    rotated_encodings = encodings @ R.T
    sorted_encodings = np.sort(rotated_encodings, axis=0)
    gaps = np.diff(sorted_encodings, axis=0)

    min_gap_per_axis = np.min(gaps, axis=0)
    global_min_gap = np.min(min_gap_per_axis)

    is_disjoint = global_min_gap > delta

    return is_disjoint, global_min_gap, min_gap_per_axis



def get_difference_vectors(encodings_tensor):
    """
    Computes all pairwise difference vectors for the dataset.
    For N=100, this returns a tensor of shape (4950, d).
    """
    n = encodings_tensor.shape[0]
    # Broadcasting to get all pairs: shape (N, N, d)
    diffs = encodings_tensor.unsqueeze(1) - encodings_tensor.unsqueeze(0)
    # Extract only the upper triangle (unique pairs where i < j)
    i, j = torch.triu_indices(n, n, offset=1)
    return diffs[i, j]


def optimize_rotation(encodings_np, delta=0.05, lr=0.05, max_epochs=15000):
    """
    Performs Lie algebra gradient descent to find a globally disjoint rotation.
    """
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running optimization on: {device}")

    # 1. Prepare the difference vectors
    encodings_tensor = torch.tensor(encodings_np, dtype=torch.float32, device=device)
    V = get_difference_vectors(encodings_tensor)
    d = V.shape[1]

    # 2. Initialize a valid random orthogonal matrix in SO(d)
    random_matrix = torch.randn(d, d, device=device)
    Q, R_qr = torch.linalg.qr(random_matrix)
    Q = Q * torch.sign(torch.diag(R_qr))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # We require gradients for R to compute the Euclidean gradient automatically
    R = Q.clone().requires_grad_(True)

    print(f"Starting optimization for {V.shape[0]} pairs in {d} dimensions...")

    # 3. The Optimization Loop
    for epoch in range(max_epochs):
        # Forward Pass: Rotate all difference vectors
        # V is (K, d), R is (d, d). Rotated V' = V @ R.T
        V_rot = torch.matmul(V, R.t())

        # Calculate the Squared Hinge Loss
        # We penalize any projection where absolute value is less than delta
        penalty = torch.relu(delta - torch.abs(V_rot))
        loss = torch.sum(penalty ** 2)

        # Check for absolute convergence
        if loss.item() == 0.0:
            print(f"✅ Converged at epoch {epoch}! Dataset is globally disjoint.")
            break

        # Backward Pass: PyTorch calculates the Euclidean gradient (G)
        loss.backward()
        G = R.grad

        # 4. Manifold Geometry Step (No gradients tracked here)
        with torch.no_grad():
            # Project Euclidean gradient onto the Lie algebra (skew-symmetric matrix A)
            A = 0.5 * (R.t() @ G - G.t() @ R)

            # Retract back to the SO(d) manifold using the Matrix Exponential
            step_matrix = torch.linalg.matrix_exp(-lr * A)

            # Update R (R_{t+1} = R_t * exp(-lr * A))
            R.copy_(torch.matmul(R, step_matrix))

            # Clear the gradients for the next iteration
            R.grad.zero_()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    if loss.item() > 0:
        print(f"⚠️ Max epochs reached. Final Loss: {loss.item():.6f}")
        print("Try increasing max_epochs, lowering delta, or adjusting the learning rate (lr).")

    return R.detach().cpu().numpy()


if __name__ == '__main__':
    encodings = np.load('lfw_encodings_100.npy')
    delta = 0.001
    optimal_R = optimize_rotation(encodings, delta=delta, lr=0.3)

    verify_globally_disjoint(encodings, optimal_R, delta=delta)