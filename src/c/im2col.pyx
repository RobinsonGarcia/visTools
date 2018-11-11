import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cpdef im2col_cython(np.ndarray[DTYPE_t, ndim=4] x, int field_height,
                  int field_width, int padding, int stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1

    cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width, N * HH * WW),
            dtype=x.dtype)

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, padding, stride)
    return cols

@cython.boundscheck(False)
cdef int im2col_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]
