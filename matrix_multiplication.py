from __future__ import division
from numba import cuda, guvectorize, float64, void, float32
import math
import numpy as np

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


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.

TPB = 32
@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x  # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@guvectorize([void(float64[:, :], float64[:, :], float64[:, :])], '(m,p),(p,n)->(m,n)', target='cuda')
def matmul_cuda(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        out[i, j] = tmp


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
    # fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    #out = matmul_cuda(A_global_mem, B_global_mem)


    # Copy the result back to the host
    return C_global_mem.copy_to_host()

