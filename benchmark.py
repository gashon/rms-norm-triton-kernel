import torch
import torch.nn.functional as F
import time
import logging

from cs336_basics.model import RMSNorm
from cs336_systems.kernel.main import (
    RMSNormTriton,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class Benchmark:
    def __init__(
        self, batch_size, last_dim_sizes, num_passes, do_backward=False, device=None
    ):
        self.batch_size = batch_size
        self.last_dim_sizes = last_dim_sizes
        self.num_passes = num_passes
        self.do_backward = do_backward
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_inputs(self, last_dim_size):
        x = torch.randn(
            self.batch_size,
            self.batch_size,
            last_dim_size,
            device=self.device,
            requires_grad=True,
        ).to("cuda")
        w = torch.ones(last_dim_size, device=self.device, requires_grad=True).to("cuda")
        dy = (
            torch.randn(
                self.batch_size, self.batch_size, last_dim_size, device=self.device
            ).to("cuda")
            if self.do_backward
            else None
        )
        return x, w, dy

    def run(self):
        for last_dim_size in self.last_dim_sizes:
            logger.info(f"\n\n--Last Dimension Size: {last_dim_size}")

            x, w, dy = self.get_inputs(last_dim_size)

            # Initialize RMSNorm
            rmsnorm = RMSNorm(last_dim_size).to(self.device)

            # Benchmark LayerNorm
            self.benchmark(
                "LayerNorm",
                lambda a: F.layer_norm(a, normalized_shape=(last_dim_size,)),
                x,
                dy,
            )

            # Benchmark RMSNorm
            self.benchmark("RMSNorm", rmsnorm, x, dy)

            # Benchmark Triton RMSNorm
            self.benchmark(
                "RMSNorm Triton",
                RMSNormTriton(last_dim_size),
                x,
                dy,
            )

            compiled_rmsnorm = torch.compile(rmsnorm)
            # Benchmark compiled rmsnorm
            self.benchmark("torch.compiled RMSNorm", compiled_rmsnorm, x, dy)

    def benchmark(self, name, func, x, dy):
        # Warm-up
        result = func(x)
        if self.do_backward and dy is not None:
            result.backward(dy)

        torch.cuda.synchronize()

        backward_time = 0
        start_time = time.time()
        for _ in range(self.num_passes):
            x.grad = None
            if isinstance(func, torch.nn.Module):
                for param in func.parameters():
                    param.grad = None
            result = func(x)
            torch.cuda.synchronize()

            if self.do_backward and dy is not None:
                start_time_backward = time.time()
                result.backward(dy)
                torch.cuda.synchronize()
                end_time_backward = time.time()
                backward_time += end_time_backward - start_time_backward
        end_time = time.time()

        duration = (end_time - start_time) * 1000 / self.num_passes

        if self.do_backward:
            backward_time = backward_time * 1000 / self.num_passes
            forward_time = duration - backward_time
            logger.info(f"{name} Backward Time (ms): {backward_time:.4f}")
            logger.info(f"{name} Forward Time (ms): {forward_time:.4f}")

        logger.info(f"{name} Total Time (ms): {duration:.4f}")


def main():
    batch_size = 250
    last_dim_sizes = [1024, 2048, 4096, 8192]
    num_passes = 1000
    do_backward = True

    benchmark = Benchmark(
        batch_size=batch_size,
        last_dim_sizes=last_dim_sizes,
        num_passes=num_passes,
        do_backward=do_backward,
    )

    benchmark.run()


if __name__ == "__main__":
    main()
