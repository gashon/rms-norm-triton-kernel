import torch


def compute_grad_g(grad_rmsnorm_L, x, g):
    """
    Computes the gradient of L with respect to g.

    Parameters:
    - grad_rmsnorm_L: The gradient of the loss L with respect to the output of RMSNorm(x, g).
    - x: The input matrix with any shape, where the last dimension matches the dimensionality of g.
    - g: The parameter vector of shape (H,).

    Returns:
    - grad_g: The gradient of the loss L with respect to g.
    """

    # Compute the denominator D
    d_model = x.size(-1)
    D = torch.sqrt(torch.sum(x**2, dim=-1) / d_model + 1e-5).unsqueeze(
        -1
    )  # Add small epsilon for numerical stability

    # Compute the Jacobian-vector product for g
    grad_g = grad_rmsnorm_L * (x / D)

    # Sum over all dimensions except for the last one to match the dimensionality of g
    grad_g = torch.sum(grad_g, dim=tuple(range(grad_g.dim() - 1)))

    return grad_g


import torch


def compute_grad_x(grad_rmsnorm_L, x, g):
    """
    Computes the gradient of L with respect to x.

    Parameters:
    - grad_rmsnorm_L: The gradient of the loss L with respect to the output of RMSNorm(x, g).
    - x: The input matrix with any shape, where the last dimension matches the dimensionality of g.
    - g: The parameter vector of shape (H,).

    Returns:
    - grad_x: The gradient of the loss L with respect to x.
    """

    # Compute the denominator D, reshaped to broadcast correctly when multiplying with x
    d_model = x.size(-1)
    D = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / d_model + 1e-5)

    # Compute the gradient of RMSNorm with respect to x
    grad_x = grad_rmsnorm_L * (g / D) - (x * grad_rmsnorm_L * g / D**3).sum(
        dim=-1, keepdim=True
    ) * (x / d_model)

    return grad_x
