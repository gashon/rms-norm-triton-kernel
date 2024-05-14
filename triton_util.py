import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X,
    Y,
    W,
    Rstd,
    stride_ml,
    stride_n,
    L,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements a forward kernel for root mean square layer normalization.

    Parameters:
    X (tl.tensor): Input tensor where each column represents a feature.
    Y (tl.tensor): Output tensor for normalized features.
    W (tl.tensor): Weights for scaling the normalized data.
    Rstd (tl.tensor): Tensor to store reciprocal of the computed standard deviations.
    stride_ml (int): Stride to access elements along the combined dimensions M and L.
    stride_n (int): Stride to access elements along dimension N.
    L (int): Size of the second dimension in the batch.
    N (int): Total number of features per instance.
    eps (float): Small epsilon value for numerical stability in division.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.

    """
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * stride_ml + batch * stride_n
    Y += base_idx
    X += base_idx

    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += a * a
    rms = tl.sqrt(tl.sum(_rms) / N + eps)

    # Store the reciprocal of the standard deviation
    tl.store(Rstd + row * L + batch, rms)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x / rms
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    input_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    grad_output_ptr: tl.pointer_type,
    input_row_stride: tl.uint32,
    grad_input_ptr: tl.pointer_type,
    grad_weight_accum_ptr: tl.pointer_type,
    num_elements: tl.uint32,
    eps: tl.float32,
    block_size: tl.constexpr,
):
    # Calculate the row index for this program instance
    row_idx = tl.program_id(0)

    # Create an array of offsets within the block
    offsets = tl.arange(0, block_size)

    # Calculate memory access ranges for the inputs and gradients
    input_offsets = row_idx * input_row_stride + offsets
    input_ptrs = input_ptr + input_offsets
    weight_ptrs = weight_ptr + offsets
    grad_output_offsets = grad_output_ptr + input_offsets

    # Create masks to handle cases where block size may exceed the number of elements
    valid_elements_mask = offsets < num_elements

    # Load input values, weights, and gradient outputs using the computed offsets and masks
    input_values = tl.load(input_ptrs, mask=valid_elements_mask, other=0)
    weights = tl.load(weight_ptrs, mask=valid_elements_mask, other=0)
    grad_outputs = tl.load(grad_output_offsets, mask=valid_elements_mask, other=0)

    # Compute the normalization factor from the input values
    norm_factor = tl.sqrt(tl.sum(input_values * input_values) / num_elements + eps)

    # Compute partial gradients with respect to weights
    grad_weight_partial = input_values * grad_outputs / norm_factor
    tl.store(
        grad_weight_accum_ptr + input_offsets,
        grad_weight_partial,
        mask=valid_elements_mask,
    )

    # Compute partial gradients with respect to input values
    grad_input_first_term = grad_outputs * weights / norm_factor
    grad_input_second_term = (
        tl.sum(input_values * grad_outputs * weights)
        * input_values
        / (num_elements * norm_factor * norm_factor * norm_factor)
    )
    grad_input_values = grad_input_first_term - grad_input_second_term
    tl.store(
        grad_input_ptr + input_offsets, grad_input_values, mask=valid_elements_mask
    )
