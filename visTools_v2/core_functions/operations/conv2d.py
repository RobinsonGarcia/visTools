import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    px,py = padding
    assert (H + 2 * py - field_height) % stride == 0
    assert (W + 2 * px - field_height) % stride == 0
    out_height = int((H + 2 * py - field_height) / stride + 1)
    out_width = int((W + 2 * px - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    px,py = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (py, py), (px, px)), mode='edge')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
    stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

"""writen as part of cs231n exercise"""
def conv2d(x, w, stride=1):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW
    - 'stride': The number of pixels between adjacent receptive fields in the
    horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ########################################################################### 

    N,C,H,W = x.shape    
    F,_,Hh,Ww = w.shape
    s = stride
    padx = int((W*(s-1)-s+Ww)/2)
    pady = int((H*(s-1)-s+Hh)/2)


    H2,W2 = H,W
    H2 = int(H2)
    W2 = int(W2)

    X_col = im2col_indices(x, field_height=Hh, field_width=Ww, padding=(padx,pady), stride=s)
   # X_col = im2col(x,params)

    W_col = np.reshape(w,(F,C*Hh*Ww))

    W_col = np.rollaxis(np.array(W_col),0,2)[np.newaxis,:,:]
    out = np.matmul(X_col.T,W_col)
    out = np.rollaxis(out,0,3)

    out = np.reshape(out,(N,F,H2,W2))
    ###########################################################################
    ###########################################################################
    return out
