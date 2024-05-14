import torch
import triton

from cs336_systems.kernel.triton_util import rmsnorm_fwd_kernel, rmsnorm_bwd_kernel


class RMSNormAutogradFuncClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, eps=1e-5):
        """
        x: input tensor of shape [M, N]
        g: scale vector of shape [N]
        eps: small constant to avoid division by zero
        """
        # Save the input tensors for the backward pass
        ctx.save_for_backward(x, g)
        ctx.eps = eps

        # Compute the squared sum and normalize
        x_squared = x * x
        x_squared_sum = x_squared.sum(dim=-1, keepdim=True)
        x_norm = torch.rsqrt(x_squared_sum / x.shape[-1] + eps)

        # Apply the normalization and scale by weight
        output = x * x_norm * g

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: gradient of the loss w.r.t. the output of the forward function
        """
        x, g = ctx.saved_tensors
        eps = ctx.eps
        N = x.size(-1)

        # Compute shared terms
        x_squared = x * x
        x_squared_sum = x_squared.sum(dim=-1, keepdim=True)
        x_norm = torch.rsqrt(x_squared_sum / N + eps)

        # Gradient w.r.t. x
        grad_x_norm = grad_output * g  # scale by g
        grad_x_part1 = grad_x_norm * x_norm  # apply normalized scaling

        grad_x_squared_sum = (-0.5 * (x_squared_sum / N + eps) ** (-1.5)) * (2 * x / N)
        grad_x_part2 = grad_x_squared_sum * (x * grad_x_norm).sum(dim=-1, keepdim=True)

        grad_x = grad_x_part1 + grad_x_part2

        # Gradient w.r.t. g
        grad_g = (grad_output * x * x_norm).sum(dim=0)

        return grad_x, grad_g, None


class RMSNormTritonAutogradFuncClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, eps=1e-5):
        M, L, N = x.shape
        y = torch.empty_like(x, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M * L, dtype=torch.float32, device=x.device)

        grid = (M, L)
        rmsnorm_fwd_kernel[grid](
            x, y, g, rstd, x.stride(0), x.stride(1), L, N, eps, BLOCK_SIZE=128
        )

        ctx.block_size = triton.next_power_of_2(N)
        ctx.save_for_backward(x, g)
        ctx.eps = eps
        ctx.N = N
        ctx.L = L

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        H = x.shape[-1]
        x_shape = x.shape

        x = x.view(-1, H)
        n_rows = x.shape[0]

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(x)

        rmsnorm_bwd_kernel[(n_rows,)](
            x,
            weight,
            grad_output,
            x.stride(0),
            grad_x,
            partial_grad_weight,
            H,
            eps,
            num_warps=16,
            block_size=ctx.block_size,
        )
        return grad_x.view(*x_shape), partial_grad_weight.sum(dim=0)


class RMSNormTriton(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(H)).to("cuda")

    def forward(self, x):
        return RMSNormTritonAutogradFuncClass.apply(x, self.weight)

    @staticmethod
    def apply(x, weight, eps=1e-5):
        return RMSNormTritonAutogradFuncClass.apply(x, weight, eps)
