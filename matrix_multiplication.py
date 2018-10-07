from __future__ import division
from numba import cuda, float32
import math


@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


def matrix_mult(m1, m2):
    A_global_mem = cuda.to_device(m1)
    B_global_mem = cuda.to_device(m2)

    # Allocate memory on the device for the result
    C_global_mem = cuda.device_array((A_global_mem.shape[0], B_global_mem.shape[1]))

    # Configure the blocks
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(m1.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(m2.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    # # Start the kernel
    matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

    # Copy the result back to the host
    return C_global_mem.copy_to_host()
