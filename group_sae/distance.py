import torch
import torch.nn.functional as F
from torch import Tensor


class Distance:
    """Base class for distance computation."""

    def __init__(self):
        """Initializes the distance storage."""
        self.dist = []

    def update(self, *args, **kwargs):
        """Abstract method to update distance metrics."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def value(self, *args, **kwargs):
        """Abstract method to return final distance value."""
        raise NotImplementedError("Subclasses must implement this method.")


class AngularDistance(Distance):
    """Computes the angular distance between two tensors."""

    def __init__(self):
        """Initializes the AngularDistance class."""
        super().__init__()

    def update(self, A: Tensor, B: Tensor):
        """
        Computes and updates the angular distance between tensors A and B.

        Args:
            A (Tensor): First input tensor of shape [N, D].
            B (Tensor): Second input tensor of shape [N, D].
        """
        if A.shape != B.shape:
            raise ValueError(
                f"Input tensors must have the same shape, got {A.shape} and {B.shape}"
            )

        cosine_similarity = F.cosine_similarity(A, B, dim=-1)  # [N]

        # Ensure numerical stability (cosine similarity should be in [-1, 1])
        cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)

        angular_distance = torch.arccos(cosine_similarity).mean()
        self.dist.append(angular_distance / torch.pi)

    def value(self) -> Tensor:
        return torch.stack(self.dist).mean()



def linear_kernel(A: Tensor) -> Tensor:
    """Computes the linear kernel (Gram matrix) for input A."""
    return A @ A.T


def rbf_kernel(A: Tensor, bandwidth: float = 1.0) -> Tensor:
    """Computes the RBF kernel (Gaussian kernel) for input A."""
    pairwise_sq_dists = torch.cdist(A, A, p=2).pow(2)
    return torch.exp(-pairwise_sq_dists / (2 * bandwidth**2))


class CKA(Distance):
    def __init__(self, kernel: str = "linear", bandwidth: float = 1.0):
        """
        Initializes the CKA class with the specified kernel.

        Args:
            kernel (str): Type of kernel to use ("linear" or "rbf").
            bandwidth (float): Bandwidth parameter for the RBF kernel (ignored for linear).
        """
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth

        if kernel == "linear":
            self.kernel_fn = linear_kernel
        elif kernel == "rbf":
            self.kernel_fn = lambda A: rbf_kernel(A, self.bandwidth)
        else:
            raise ValueError("Invalid kernel type. Choose 'linear' or 'rbf'.")

    def hisc_0(self, K: Tensor, L: Tensor) -> Tensor:
        """
        Computes the Hilbert-Schmidt Independence Criterion (HSIC).

        Args:
            K (Tensor): Kernel matrix for A.
            L (Tensor): Kernel matrix for B.

        Returns:
            Tensor: HSIC score.
        """
        n = K.size(0)
        H = (
            torch.eye(n, device=K.device, dtype=K.dtype)
            - torch.ones((n, n), device=K.device, dtype=K.dtype) / n
        )
        return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)

    def update(self, A: Tensor, B: Tensor):
        """
        Computes and updates the Centered Kernel Alignment (CKA) distance.

        Args:
            A (Tensor): First input tensor [N, D].
            B (Tensor): Second input tensor [N, D].
        """
        K = self.kernel_fn(A)
        L = self.kernel_fn(B)

        cross = self.hisc_0(K, L)
        self_K = self.hisc_0(K, K)
        self_L = self.hisc_0(L, L)

        cka = cross / torch.sqrt(
            self_K * self_L + 1e-8
        )  # Added epsilon for numerical stability
        self.dist.append(1 - cka)

    def value(self) -> Tensor:
        return torch.stack(self.dist).mean()


class ApproxCKA(Distance):
    """
    Approximate Centered Kernel Alignment (CKA) computation using the HSIC_1 estimator.

    Reference:
    Nguyen, Thao, Maithra Raghu, and Simon Kornblith.
    "Do wide and deep networks learn the same things? Uncovering how neural network
    representations vary with width and depth." arXiv preprint arXiv:2010.15327 (2020).
    """

    def __init__(self, kernel: str = "linear", bandwidth: float = 1.0):
        """
        Initializes the ApproxCKA class with the specified kernel.

        Args:
            kernel (str): Type of kernel to use ("linear" or "rbf").
            bandwidth (float): Bandwidth parameter for the RBF kernel (ignored for linear).
        """
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth

        # Storage for HSIC_1 values
        self.cross = []
        self.self_K = []
        self.self_L = []

        # Set kernel function
        if kernel == "linear":
            self.kernel_fn = linear_kernel
        elif kernel == "rbf":
            self.kernel_fn = lambda A: rbf_kernel(A, self.bandwidth)
        else:
            raise ValueError("Invalid kernel type. Choose 'linear' or 'rbf'.")

    def hisc_1(self, K: Tensor, L: Tensor) -> Tensor:
        """
        Computes the HSIC_1 estimator for kernel matrices K and L.

        Args:
            K (Tensor): Kernel matrix for the first representation (shape: [N, N]).
            L (Tensor): Kernel matrix for the second representation (shape: [N, N]).

        Returns:
            Tensor: Approximated HSIC value.
        """
        n = K.size(0)
        if n < 4:
            raise ValueError(f"Number of samples must be at least 4, got {n}.")

        # Create identity and ones matrices
        eye = torch.eye(n, dtype=K.dtype, device=K.device)
        one = torch.ones((n, 1), dtype=K.dtype, device=K.device)

        # Zero out diagonal entries of K and L
        K_tilde = K * (1 - eye)
        L_tilde = L * (1 - eye)

        # Compute HSIC_1 terms
        term1 = (K_tilde @ L_tilde).diag().sum()  # trace(\tilde{K} \tilde{L})
        term2 = (one.T @ K_tilde @ one @ one.T @ L_tilde @ one) / ((n - 1) * (n - 2))
        term3 = (one.T @ K_tilde @ L_tilde @ one) * (2.0 / (n - 2))

        # HSIC_1 formula
        hsic_vals = (term1 + term2 - term3) / (n * (n - 3))
        return hsic_vals

    def update(self, A: Tensor, B: Tensor):
        """
        Computes and stores the approximated CKA distance using the HSIC_1 estimator.

        Args:
            A (Tensor): First input tensor [N, D].
            B (Tensor): Second input tensor [N, D].
        """
        if A.shape != B.shape:
            raise ValueError(f"Input tensors must have the same shape, got {A.shape} and {B.shape}")

        K = self.kernel_fn(A)
        L = self.kernel_fn(B)

        # Store HSIC_1 estimates
        self.cross.append(self.hisc_1(K, L))
        self.self_K.append(self.hisc_1(K, K))
        self.self_L.append(self.hisc_1(L, L))

    def value(self) -> Tensor:
        """
        Computes the final CKA score from stored HSIC_1 values.

        Returns:
            Tensor: The approximated CKA similarity measure.
        """
        if not self.cross:
            raise RuntimeError("No stored HSIC_1 values. Call `update()` before computing `value()`.")

        cross = torch.stack(self.cross).mean()
        self_K = torch.stack(self.self_K).mean().sqrt()
        self_L = torch.stack(self.self_L).mean().sqrt()

        return 1 - cross / (self_K * self_L + 1e-8)  # Avoid division by zero
    

class SVCCA(Distance):
    """
    Singular Vector Canonical Correlation Analysis (SVCCA).
    
    This method measures the similarity between two sets of representations
    based on the principal angles between their subspaces using Singular Value Decomposition (SVD).
    
    Reference:
    - Golub, Gene H., and Charles F. Van Loan. "Matrix Computations." 3rd edition, Johns Hopkins University Press, 1996.
    """

    def __init__(self, top_k: int = None):
        """
        Initializes the SVCAA class.

        Args:
            top_k (int, optional): Number of top singular vectors to use.
                                   If None, uses all available singular vectors.
        """
        super().__init__()
        self.top_k = top_k

    def canonical_angles(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Computes the principal angles between the subspaces spanned by A and B.

        Args:
            A (Tensor): First input tensor [N, D].
            B (Tensor): Second input tensor [N, D].

        Returns:
            Tensor: Cosines of the principal angles between subspaces.
        """
        if A.shape != B.shape:
            raise ValueError(f"Input tensors must have the same shape, got {A.shape} and {B.shape}")

        # Compute SVD of both matrices
        U_A, _, _ = torch.linalg.svd(A, full_matrices=False)
        U_B, _, _ = torch.linalg.svd(B, full_matrices=False)

        # Optionally truncate to top_k singular vectors
        if self.top_k is not None:
            U_A = U_A[:, :self.top_k]
            U_B = U_B[:, :self.top_k]

        # Compute singular values of U_A^T * U_B
        S = torch.linalg.svdvals(U_A.T @ U_B)

        # Clamp values to avoid numerical instability issues (should be in [0, 1])
        S = torch.clamp(S, 0, 1)

        return torch.acos(S)  # Principal angles in radians

    def update(self, A: Tensor, B: Tensor):
        """
        Computes and stores the similarity measure using SVCAA.

        Args:
            A (Tensor): First input tensor [N, D].
            B (Tensor): Second input tensor [N, D].
        """
        angles = self.canonical_angles(A, B)

        # Compute mean of squared cosines (a common similarity measure)
        sv_similarity = torch.mean(torch.cos(angles).pow(2))
        self.dist.append(1 - sv_similarity)  # Distance measure

    def value(self) -> Tensor:
        """
        Computes the final SVCAA similarity score.

        Returns:
            Tensor: The SVCAA similarity measure.
        """
        if not self.dist:
            raise RuntimeError("No stored values. Call `update()` before computing `value()`.")

        return torch.tensor(self.dist).mean()