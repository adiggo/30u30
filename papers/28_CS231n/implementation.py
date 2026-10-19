"""
implementation.py - CNN from Scratch in NumPy (forward and backward)

Implements the core convolutional neural network layers described in
Stanford CS231n (https://cs231n.github.io/convolutional-networks/):
  - Convolutional layer (naive + im2col), forward and backward (col2im)
  - Max pooling layer, forward and backward (gradient routed to the max)
  - ReLU activation (+ backward)
  - Fully connected layer (+ backward)
  - Softmax / cross-entropy loss (+ gradient with respect to the scores)
  - SimpleCNN class (CONV-RELU-POOL-FC pipeline) with forward, backward and
    an SGD-with-momentum update, which train_minimal.py uses to train it

[PATCH 2026-10-18] The backward functions, SimpleCNN.backward/update and the
vectorized im2col/col2im are new. Every backward is checked against
torch.autograd (float64) and centred finite differences in
validation/day28_backprop_check.py. Convention: a backward function takes the
upstream gradient `dout` plus the forward pass's inputs, and returns the
gradients in the order of the forward's arguments (dx, dw, db).

Usage:
    python implementation.py
    python implementation.py --demo

Reference: CS231n Course Notes - https://cs231n.github.io/convolutional-networks/
"""

import numpy as np
import argparse


# ---------------------------------------------------------------------------
# Convolutional Layer
# ---------------------------------------------------------------------------

def conv_forward_naive(x, w, b, stride=1, pad=0):
    """
    Naive convolution forward pass using nested loops.

    Slides each filter across the input volume and computes dot products.
    Output size per CS231n: (W - F + 2P) / S + 1

    Args:
        x: Input volume, shape (N, C, H, W) — batch, channels, height, width
        w: Filters, shape (K, C, FH, FW) — num_filters, channels, filter_h, filter_w
        b: Biases, shape (K,)
        stride: Step size for sliding the filter
        pad: Zero-padding added to input borders

    Returns:
        out: Output volume, shape (N, K, H_out, W_out)
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape

    # Output size formula from CS231n
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    out = np.zeros((N, K, H_out, W_out))

    for n in range(N):                  # Each image in batch
        for k in range(K):              # Each filter
            for i in range(H_out):      # Each output row
                for j in range(W_out):  # Each output col
                    h_start = i * stride
                    w_start = j * stride
                    # Extract local region and compute dot product
                    # This is exactly the operation CS231n describes:
                    # "compute dot products between the entries of the filter
                    # and the input at any position"
                    region = x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                    out[n, k, i, j] = np.sum(region * w[k]) + b[k]

    return out


def im2col_naive(x, FH, FW, stride=1, pad=0):
    """
    Reshape input patches into columns for matrix-multiply convolution
    (readable loop version).

    CS231n describes this trick:
    "The local regions in the input image are stretched out into columns
    in an operation commonly called im2col."

    [PATCH 2026-10-18] This is the original loop implementation, renamed from
    `im2col` and kept because it shows plainly what im2col computes. `im2col`
    below builds the identical matrix without Python loops, which training needs.

    Args:
        x: Input, shape (N, C, H, W)
        FH, FW: Filter height and width
        stride: Stride
        pad: Padding

    Returns:
        cols: Columns matrix, shape (C*FH*FW, N*H_out*W_out). Column
              n*H_out*W_out + i*W_out + j is the flattened (C, FH, FW) patch
              under output position (i, j) of image n.
    """
    N, C, H, W = x.shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    cols = np.zeros((C * FH * FW, N * H_out * W_out))

    col_idx = 0
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                cols[:, col_idx] = patch.ravel()  # ravel: (C, FH, FW) -> (C*FH*FW,), row-major: c slowest, fw fastest
                col_idx += 1

    return cols


def im2col_indices(x_shape, FH, FW, stride=1, pad=0):
    """
    [PATCH 2026-10-18] Index arrays that address every im2col entry in the padded input.

    Entry (r, q) of one image's im2col matrix is

        x_padded[n, c, i*stride + fh, j*stride + fw]

    where the row r stands for (c, fh, fw) and the column q for the output
    position (i, j), both flattened row-major (NumPy's default "C" order, last
    index fastest): r = c*FH*FW + fh*FW + fw and q = i*W_out + j. That is the
    order of patch.ravel() and of the (i, j) loops in im2col_naive. With the three arrays returned here,
    x_padded[n, c_idx, h_idx, w_idx] is that whole (C*FH*FW, H_out*W_out)
    matrix in one indexing operation.

    Args:
        x_shape: (N, C, H, W) of the unpadded input
        FH, FW, stride, pad: as in im2col

    Returns:
        c_idx: (C*FH*FW, 1) channel of each row (the same in every column)
        h_idx: (C*FH*FW, H_out*W_out) row of each entry in the padded input
        w_idx: (C*FH*FW, H_out*W_out) column of each entry in the padded input
    """
    N, C, H, W = x_shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    # Row r <-> (c, fh, fw). Example for C=2, FH=FW=2 (8 rows):
    #   c_idx  = [0 0 0 0 1 1 1 1]   np.repeat(a, k): every element of a, k times in a row
    #   fh_idx = [0 0 1 1 0 0 1 1]   np.tile(a, k): the whole of a, k times end to end
    #   fw_idx = [0 1 0 1 0 1 0 1]
    c_idx = np.repeat(np.arange(C), FH * FW)            # (C*FH*FW,)
    fh_idx = np.tile(np.repeat(np.arange(FH), FW), C)   # (C*FH*FW,) offset inside the window, down
    fw_idx = np.tile(np.arange(FW), FH * C)             # (C*FH*FW,) offset inside the window, across

    # Column q <-> output position (i, j), i slowest: i_idx = [0 .. 0 1 .. 1 ..], j_idx = [0 1 .. W_out-1 0 1 ..]
    i_idx = np.repeat(np.arange(H_out), W_out)          # (H_out*W_out,)
    j_idx = np.tile(np.arange(W_out), H_out)            # (H_out*W_out,)

    # Pixel = window start (i*stride) + offset inside the window (fh). a[:, None] turns a into a column
    # (one value per matrix row) and b[None, :] into a row (one value per matrix column); adding a
    # (rows, 1) array to a (1, cols) array broadcasts to the full (rows, cols) grid of sums.
    h_idx = fh_idx[:, None] + stride * i_idx[None, :]   # (C*FH*FW, H_out*W_out)
    w_idx = fw_idx[:, None] + stride * j_idx[None, :]   # (C*FH*FW, H_out*W_out)
    return c_idx[:, None], h_idx, w_idx


def im2col(x, FH, FW, stride=1, pad=0):
    """
    Reshape input patches into columns for matrix-multiply convolution.

    CS231n describes this trick:
    "The local regions in the input image are stretched out into columns
    in an operation commonly called im2col."

    [PATCH 2026-10-18] Vectorized: the same matrix as im2col_naive (identical
    values, max |diff| = 0), gathered with one fancy-indexing operation instead
    of N*H_out*W_out Python iterations (about 10x faster on a CIFAR batch).

    Args:
        x: Input, shape (N, C, H, W)
        FH, FW: Filter height and width
        stride: Stride
        pad: Padding

    Returns:
        cols: Columns matrix, shape (C*FH*FW, N*H_out*W_out), columns in the
              (n, i, j) order of im2col_naive. Same dtype as x.
    """
    N, C, H, W = x.shape
    c_idx, h_idx, w_idx = im2col_indices(x.shape, FH, FW, stride, pad)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # Fancy (integer-array) indexing: c_idx (CFF, 1), h_idx and w_idx (CFF, HW) broadcast to (CFF, HW),
    # and entry (r, q) reads the single pixel x_padded[n, c_idx[r], h_idx[r, q], w_idx[r, q]]. The
    # leading ':' keeps the batch axis, so the result is (N, C*FH*FW, H_out*W_out). It is a copy, and a
    # pixel that lies in several (overlapping) windows is simply read several times.
    cols = x_padded[:, c_idx, h_idx, w_idx]
    # Make the columns run (n, i, j) like im2col_naive: transpose(1, 0, 2) swaps the first two axes,
    # (N, CFF, HW) -> (CFF, N, HW); reshape then merges (N, HW) into one axis of N*HW (n slowest).
    return cols.transpose(1, 0, 2).reshape(C * FH * FW, -1)


def conv_forward_im2col(x, w, b, stride=1, pad=0, return_cols=False):
    """
    Convolution via im2col + matrix multiply.

    CS231n: "The result of a convolution is now equivalent to performing
    one large matrix multiply np.dot(W_row, X_col)."

    This is much faster than the naive version because it leverages
    optimized BLAS routines for the matrix multiply.

    Args:
        x: Input volume, shape (N, C, H, W)
        w: Filters, shape (K, C, FH, FW)
        b: Biases, shape (K,)
        stride, pad: As in conv_forward_naive
        return_cols: [PATCH 2026-10-18] if True, also return the im2col
            matrix; the backward pass needs it for dw (see conv_backward_im2col)

    Returns:
        out: Output volume, shape (N, K, H_out, W_out)
        cols: (only if return_cols) shape (C*FH*FW, N*H_out*W_out)
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    # Stretch input patches into columns
    cols = im2col(x, FH, FW, stride, pad)

    # Reshape filters into rows: (K, C*FH*FW)
    w_row = w.reshape(K, -1)

    # Matrix multiply: (K, C*FH*FW) @ (C*FH*FW, N*H_out*W_out) = (K, N*H_out*W_out)
    out = w_row @ cols + b.reshape(K, 1)

    # Reshape back to (N, K, H_out, W_out)
    out = out.reshape(K, N, H_out, W_out).transpose(1, 0, 2, 3)

    if return_cols:  # [PATCH 2026-10-18] hand X_col to the backward pass instead of rebuilding it
        return out, cols
    return out


# ---------------------------------------------------------------------------
# [PATCH 2026-10-18] Convolution backward: col2im and conv_backward
# ---------------------------------------------------------------------------
#
# In the im2col view the forward pass is one matrix multiply (CS231n,
# "Implementation as Matrix Multiplication"):
#     out_row = W_row @ X_col + b          out_row: (K, N*H_out*W_out)
# With dout_row = dL/d(out_row) in the same layout, the usual matrix-multiply
# gradients (CS231n optimization-2, "Gradients for vectorized operations") give
#     db     = dout_row summed over its columns               (K,)
#     dW_row = dout_row @ X_col.T                             (K, C*FH*FW)
#     dX_col = W_row.T @ dout_row                             (C*FH*FW, N*H_out*W_out)
# and col2im turns dX_col back into dx by adding every column onto the input
# patch it was copied from. A pixel inside several overlapping windows was
# copied into several columns, so its gradient is the SUM of those entries:
# "gradients add up at forks ... use += instead of =" (CS231n optimization-2,
# "Backprop in practice: Staged computation"). col2im therefore accumulates.

def col2im_naive(cols, x_shape, FH, FW, stride=1, pad=0):
    """
    [PATCH 2026-10-18] Inverse direction of im2col_naive, with accumulation (loop version).

    Adds column q of `cols` onto the (C, FH, FW) input patch that im2col
    copied into column q. Why this is the right backward: im2col is linear,
    X_col = M x for a big 0/1 matrix M that copies pixels into columns, so
    the chain rule gives dx = M.T @ dX_col, the same rule as dx = W.T @ dy for
    y = W x. Multiplying by M.T sends every entry of dX_col back to the pixel
    it was copied from and sums what arrives at the same pixel. That is
    col2im; M.T is called the transpose or "adjoint" of im2col.

    Args:
        cols: shape (C*FH*FW, N*H_out*W_out), e.g. dX_col
        x_shape: (N, C, H, W) of the unpadded input
        FH, FW, stride, pad: as in im2col

    Returns:
        x: shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1

    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad))

    col_idx = 0
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                # reshape(C, FH, FW) undoes patch.ravel(); '+=' (not '=') because overlapping windows
                # share pixels, and each pixel must collect the contributions of all its windows
                x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW] += cols[:, col_idx].reshape(C, FH, FW)
                col_idx += 1

    # Drop the padding border: what landed on padded zeros belongs to no input pixel.
    # (pad:pad + H also works for pad = 0, where [pad:-pad] would give an empty array.)
    return x_padded[:, :, pad:pad + H, pad:pad + W]


def col2im(cols, x_shape, FH, FW, stride=1, pad=0):
    """
    [PATCH 2026-10-18] Inverse direction of im2col, with accumulation (vectorized).

    Same result as col2im_naive, using the index arrays of im2col_indices and
    a scatter-add (np.add.at) instead of Python loops.

    Args:
        cols: shape (C*FH*FW, N*H_out*W_out), e.g. dX_col
        x_shape: (N, C, H, W) of the unpadded input
        FH, FW, stride, pad: as in im2col

    Returns:
        x: shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W - FW + 2 * pad) // stride + 1
    c_idx, h_idx, w_idx = im2col_indices(x_shape, FH, FW, stride, pad)

    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    # Undo im2col's last two steps: reshape (CFF, N*HW) -> (CFF, N, HW), then transpose(1, 0, 2) -> (N, CFF, HW),
    # the shape that x_padded[:, c_idx, h_idx, w_idx] has.
    cols_n = cols.reshape(C * FH * FW, N, H_out * W_out).transpose(1, 0, 2)
    # np.add.at(a, index, values) performs a[index] += values *unbuffered*: an element named several times by
    # the index (a pixel shared by overlapping windows) receives every one of its values. The plain
    # x_padded[:, c_idx, h_idx, w_idx] += cols_n is buffered: it reads all targets, adds, and writes them back
    # once, so for a repeated index only one contribution survives and the gradient comes out wrong.
    # slice(None) is how you write ':' (all images) inside an index tuple.
    np.add.at(x_padded, (slice(None), c_idx, h_idx, w_idx), cols_n)

    return x_padded[:, :, pad:pad + H, pad:pad + W]    # drop the padding border


def conv_backward_naive(dout, x, w, stride=1, pad=0):
    """
    [PATCH 2026-10-18] Naive convolution backward pass: the forward's four loops, run backwards.

    Forward: out[n, k, i, j] = sum(x_padded[n, :, window(i, j)] * w[k]) + b[k].
    Each output is a dot product, and a multiply gate "swaps" its inputs
    (CS231n optimization-2, "Patterns in backward flow"): the output's gradient
    g = dout[n, k, i, j] sends g * w[k] to its window of the input and
    g * window to w[k]. All positions share w[k], so dw[k] adds up the
    contributions of every position and image: "these gradients will be added
    up across each depth slice" (CS231n, Convolutional Layer, parameter
    sharing). For stride 1, the dx built this way equals a convolution of dout
    with the spatially flipped filters, which is the notes' "Backpropagation"
    remark in that section (validation/day28_backprop_check.py checks it).

    Args:
        dout: Upstream gradient dL/dout, shape (N, K, H_out, W_out)
        x: Forward input, shape (N, C, H, W)
        w: Filters, shape (K, C, FH, FW)
        stride, pad: As in the forward pass (b is not needed)

    Returns:
        dx: shape (N, C, H, W)
        dw: shape (K, C, FH, FW)
        db: shape (K,)
    """
    N, C, H, W = x.shape
    K, _, FH, FW = w.shape
    _, _, H_out, W_out = dout.shape

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros(x_padded.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(K)

    for n in range(N):
        for k in range(K):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    g = dout[n, k, i, j]                    # gradient of this one output number
                    db[k] += g                              # b[k] was added to every output of filter k
                    dw[k] += g * x_padded[n, :, h_start:h_start + FH, w_start:w_start + FW]
                    dx_padded[n, :, h_start:h_start + FH, w_start:w_start + FW] += g * w[k]  # += : windows overlap

    return dx_padded[:, :, pad:pad + H, pad:pad + W], dw, db


def conv_backward_im2col(dout, x, w, stride=1, pad=0, cols=None, need_dx=True):
    """
    [PATCH 2026-10-18] Convolution backward pass via im2col: two matrix multiplies and a col2im.

    See the comment block above for the formulas. Same result as
    conv_backward_naive (and as torch.autograd through F.conv2d).

    Args:
        dout: Upstream gradient, shape (N, K, H_out, W_out)
        x: Forward input, shape (N, C, H, W)
        w: Filters, shape (K, C, FH, FW)
        stride, pad: As in the forward pass
        cols: im2col(x) from the forward pass (conv_forward_im2col(...,
            return_cols=True)); recomputed if None
        need_dx: if False, skip the input gradient (and its col2im). The first
            layer's input is the image, which needs no gradient for training.

    Returns:
        dx: shape (N, C, H, W), or None if need_dx is False
        dw: shape (K, C, FH, FW)
        db: shape (K,)
    """
    K, C, FH, FW = w.shape
    if cols is None:
        cols = im2col(x, FH, FW, stride, pad)                # (C*FH*FW, N*H_out*W_out)

    # The forward reshaped out_row (K, N*H_out*W_out) to (K, N, H_out, W_out) and then swapped the first
    # two axes. Undo both: transpose(1, 0, 2, 3) gives (K, N, H_out, W_out), and reshape(K, -1) gives
    # (K, N*H_out*W_out) with columns in the same (n, i, j) order as cols. (-1 = "whatever size fits".)
    dout_row = dout.transpose(1, 0, 2, 3).reshape(K, -1)

    db = dout_row.sum(axis=1)                                # b[k] was added to every column of row k
    # (K, NHW) @ (NHW, CFF) -> (K, CFF): the sum over all positions and images happens inside the matmul.
    # CS231n's "use dimension analysis" tip: this is the only product of dout_row and cols with w's shape.
    dw = (dout_row @ cols.T).reshape(w.shape)

    dx = None
    if need_dx:
        dcols = w.reshape(K, -1).T @ dout_row                # (CFF, K) @ (K, NHW) -> (CFF, NHW) = dX_col
        dx = col2im(dcols, x.shape, FH, FW, stride, pad)     # scatter-add each column onto its patch
    return dx, dw, db


# ---------------------------------------------------------------------------
# Pooling Layer
# ---------------------------------------------------------------------------

def pool_forward(x, pool_size=2, stride=2):
    """
    Max pooling forward pass.

    CS231n: "The most common form is a pooling layer with filters of size
    2x2 applied with a stride of 2 downsamples every depth slice in the
    input by 2 along both width and height, discarding 75% of the
    activations."

    Output size: (W - F) / S + 1 (no padding for pooling)

    Args:
        x: Input, shape (N, C, H, W)
        pool_size: Spatial extent of pooling window
        stride: Step size

    Returns:
        out: Pooled output, shape (N, C, H_out, W_out)
    """
    N, C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = x[:, :, h_start:h_start + pool_size, w_start:w_start + pool_size]
            out[:, :, i, j] = np.max(region, axis=(2, 3))

    return out


def pool_backward(dout, x, pool_size=2, stride=2):
    """
    [PATCH 2026-10-18] Max pooling backward pass.

    CS231n (Pooling Layer, "Backpropagation"): "the backward pass for a
    max(x, y) operation has a simple interpretation as only routing the
    gradient to the input that had the highest value in the forward pass."
    So in each window the input that won the max receives that window's
    upstream gradient and the others receive 0. The winners (the notes call
    their indices "the switches") are recomputed here from the forward input x.

    Ties: np.argmax returns the FIRST maximum in row-major window order.
    PyTorch's CPU max_pool2d also keeps the first maximum (it replaces the
    running max only on a strictly larger value), so the two agree exactly,
    ties included (validation/day28_backprop_check.py). A mask built as
    (window == window.max()) would instead copy the gradient to every tied
    input. In SimpleCNN, pooling follows a ReLU, so ties are common (windows of
    zeros); there the choice changes no gradient, because relu_backward then
    zeroes the gradient of every non-positive pre-activation anyway.

    Overlapping windows (stride < pool_size): an input can win several
    windows, and its gradients add up ('+=' below), as in exercise 2's solution.

    Args:
        dout: Upstream gradient, shape (N, C, H_out, W_out)
        x: Forward input of the pooling layer, shape (N, C, H, W)
        pool_size, stride: As in pool_forward

    Returns:
        dx: Gradient with respect to x, shape (N, C, H, W)
    """
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    P = pool_size

    dx = np.zeros(x.shape)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            # All images and channels at once: the (N, C, P, P) window, flattened row-major to (N, C, P*P)
            window = x[:, :, h_start:h_start + P, w_start:w_start + P].reshape(N, C, P * P)
            winner = np.argmax(window, axis=2)            # (N, C): flat index of the first max in each window
            # The max-pool mask, one-hot over the P*P window positions: np.arange(P*P) has shape (P*P,) and
            # winner[:, :, None] has shape (N, C, 1); '==' broadcasts them to (N, C, P*P), True only at the winner.
            mask = np.arange(P * P) == winner[:, :, None]
            # Route this window's gradient: dout[:, :, i, j] is (N, C), [:, :, None] makes it (N, C, 1) so it
            # broadcasts over the P*P positions; multiplying by the boolean mask keeps it only at the winner.
            # '+=' because with overlapping windows another window may already have routed gradient here.
            routed = mask * dout[:, :, i, j][:, :, None]          # (N, C, P*P)
            dx[:, :, h_start:h_start + P, w_start:w_start + P] += routed.reshape(N, C, P, P)

    return dx


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

def relu(x):
    """
    ReLU activation: max(0, x).

    CS231n: "RELU layer will apply an elementwise activation function,
    such as the max(0,x) thresholding at zero."
    """
    return np.maximum(0, x)


def relu_backward(dout, x):
    """
    [PATCH 2026-10-18] ReLU backward: d max(0, x)/dx is 1 where x > 0 and 0 elsewhere.

    At x == 0 exactly the derivative is taken as 0, as PyTorch does.

    Args:
        dout: Upstream gradient, same shape as x
        x: The forward INPUT of relu (the pre-activation)

    Returns:
        dx: same shape as x
    """
    return dout * (x > 0)   # (x > 0) is a boolean array; multiplying by it keeps dout where the unit was active


# ---------------------------------------------------------------------------
# Fully Connected Layer
# ---------------------------------------------------------------------------

def fc_forward(x, w, b):
    """
    Fully connected (dense) layer forward pass.

    CS231n: "Each neuron in this layer will be connected to all the
    numbers in the previous volume."

    Args:
        x: Input, shape (N, D) where D = flattened volume size
        w: Weights, shape (D, M)
        b: Biases, shape (M,)

    Returns:
        out: shape (N, M)
    """
    return x @ w + b


def fc_backward(dout, x, w):
    """
    [PATCH 2026-10-18] Fully connected backward pass for out = x @ w + b.

    CS231n optimization-2 ("Gradients for vectorized operations", "use
    dimension analysis"): dw must have w's shape (D, M) and come from x (N, D)
    and dout (N, M); the only product that fits is x.T @ dout.

    Args:
        dout: Upstream gradient, shape (N, M)
        x: Forward input, shape (N, D)
        w: Weights, shape (D, M)

    Returns:
        dx: shape (N, D)
        dw: shape (D, M)
        db: shape (M,)
    """
    dx = dout @ w.T          # (N, M) @ (M, D) -> (N, D)
    dw = x.T @ dout          # (D, N) @ (N, M) -> (D, M): the sum over the batch of each example's outer product
    db = dout.sum(axis=0)    # b was broadcast to every row (example), so its gradient sums over the rows
    return dx, dw, db


# ---------------------------------------------------------------------------
# Softmax and Cross-Entropy Loss
# ---------------------------------------------------------------------------

def softmax(scores):
    """Numerically stable softmax."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    """
    Cross-entropy loss for classification.

    Args:
        probs: Softmax probabilities, shape (N, C)
        y: True labels, shape (N,) — integer class indices

    Returns:
        loss: Scalar average cross-entropy loss
    """
    N = probs.shape[0]
    # [PATCH 2026-10-18] was -log(p + 1e-12). That is not the function whose gradient is (p - onehot)/N
    # (softmax_cross_entropy_backward): at He init some CIFAR images get p_y ~ 1e-7, where the 1e-12 shifts
    # the gradient by 1e-12 / p_y ~ 1e-5 relative and shows up as a false error in gradient checks.
    # np.maximum(p, tiny) changes nothing unless p_y underflows below ~2.2e-308 (e.g. to 0), where it keeps
    # the log finite; otherwise this is exactly -log(p_y).
    p_true = probs[np.arange(N), y]                      # fancy indexing: the true class's probability per row
    log_probs = -np.log(np.maximum(p_true, np.finfo(np.float64).tiny))
    return np.mean(log_probs)


def softmax_cross_entropy_backward(probs, y):
    """
    [PATCH 2026-10-18] Gradient of the mean cross-entropy loss with respect to the scores.

    For L = -(1/N) * sum_n log softmax(s_n)[y_n] the gradient is
        dL/ds_n = (p_n - onehot(y_n)) / N
    (CS231n neural-networks-case-study, "Computing the Analytic Gradient":
    dscores = probs; dscores[range(N), y] -= 1; dscores /= N).
    This is the exact gradient of cross_entropy_loss (whose underflow guard
    acts only if p_y is below ~2.2e-308).

    Args:
        probs: Softmax probabilities, shape (N, C)
        y: True labels, shape (N,)

    Returns:
        dscores: shape (N, C)
    """
    N = probs.shape[0]
    dscores = probs.copy()             # copy, so the caller's probs are not modified in place
    dscores[np.arange(N), y] -= 1      # fancy indexing (row n, column y[n]): subtract the one-hot target
    return dscores / N                 # the loss is a mean over the batch


# ---------------------------------------------------------------------------
# SimpleCNN: A minimal ConvNet for CIFAR-10
# ---------------------------------------------------------------------------

class SimpleCNN:
    """
    A minimal CNN following the CS231n architecture pattern:
    INPUT -> CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FC -> RELU -> FC

    Architecture (for 32x32x3 CIFAR-10 input):
      CONV1: 8 filters, 3x3, stride 1, pad 1  -> [32x32x8]
      RELU
      POOL: 2x2, stride 2                      -> [16x16x8]
      CONV2: 16 filters, 3x3, stride 1, pad 1  -> [16x16x16]
      RELU
      POOL: 2x2, stride 2                      -> [8x8x16]
      FC1: 8*8*16=1024 -> 64
      RELU
      FC2: 64 -> 10 (num classes)

    This follows the CS231n pattern:
    INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
    """

    def __init__(self, num_classes=10, input_channels=3):
        """Initialize weights using He initialization (variance = 2/fan_in)."""
        self.params = {}

        # CONV1: 8 filters, 3x3
        fan_in = input_channels * 3 * 3
        self.params['W1'] = np.random.randn(8, input_channels, 3, 3) * np.sqrt(2.0 / fan_in)
        self.params['b1'] = np.zeros(8)

        # CONV2: 16 filters, 3x3
        fan_in = 8 * 3 * 3
        self.params['W2'] = np.random.randn(16, 8, 3, 3) * np.sqrt(2.0 / fan_in)
        self.params['b2'] = np.zeros(16)

        # FC1: 1024 -> 64
        fan_in = 8 * 8 * 16
        self.params['W3'] = np.random.randn(fan_in, 64) * np.sqrt(2.0 / fan_in)
        self.params['b3'] = np.zeros(64)

        # FC2: 64 -> num_classes
        fan_in = 64
        self.params['W4'] = np.random.randn(64, num_classes) * np.sqrt(2.0 / fan_in)
        self.params['b4'] = np.zeros(num_classes)

        # [PATCH 2026-10-18] state for backprop and the momentum update
        self.cache = None      # intermediates of the last forward pass, read by backward()
        self.velocity = {}     # momentum buffers, one per parameter (created on the first update)

    def forward(self, x):
        """
        Forward pass through the network.

        [PATCH 2026-10-18] Also stores the intermediates that backward() needs
        in self.cache (CS231n's "cache"): every layer's input and every ReLU's
        pre-activation.

        Args:
            x: Input images, shape (N, 3, 32, 32)

        Returns:
            scores: Class scores, shape (N, num_classes)
        """
        p = self.params
        N = x.shape[0]

        # CONV1 -> RELU -> POOL
        a1, cols1 = conv_forward_im2col(x, p['W1'], p['b1'], stride=1, pad=1, return_cols=True)  # (N, 8, 32, 32)
        r1 = relu(a1)
        h1 = pool_forward(r1, pool_size=2, stride=2)                                               # (N, 8, 16, 16)

        # CONV2 -> RELU -> POOL
        a2, cols2 = conv_forward_im2col(h1, p['W2'], p['b2'], stride=1, pad=1, return_cols=True)  # (N, 16, 16, 16)
        r2 = relu(a2)
        h2 = pool_forward(r2, pool_size=2, stride=2)                                               # (N, 16, 8, 8)

        # Flatten: reshape (N, 16, 8, 8) -> (N, 1024), row-major (channel slowest, then row, then column)
        h2_flat = h2.reshape(N, -1)

        # FC1 -> RELU
        a3 = fc_forward(h2_flat, p['W3'], p['b3'])                                                 # (N, 64)
        h3 = relu(a3)

        # FC2 (scores)
        scores = fc_forward(h3, p['W4'], p['b4'])                                                  # (N, 10)

        self.cache = dict(x=x, cols1=cols1, a1=a1, r1=r1, h1=h1, cols2=cols2, a2=a2, r2=r2,
                          h2_shape=h2.shape, h2_flat=h2_flat, a3=a3, h3=h3)
        return scores

    def backward(self, dscores, return_dx=False):
        """
        [PATCH 2026-10-18] Backpropagation through the network.

        Walks the forward pass in reverse, applying each layer's backward
        function to the gradient coming from the layer above (CS231n
        optimization-2: the chain rule, one gate at a time). Uses the cache of
        the most recent forward() call.

        Args:
            dscores: dL/dscores, shape (N, num_classes), e.g. from
                softmax_cross_entropy_backward
            return_dx: also return dL/dx for the input images (only gradient
                checks need it; training does not, so it is skipped by default)

        Returns:
            grads: dict with the same keys and shapes as self.params
            dx: (only if return_dx) shape (N, 3, 32, 32)
        """
        if self.cache is None:
            raise RuntimeError("call forward() before backward()")
        c, p = self.cache, self.params
        grads = {}

        # FC2: scores = h3 @ W4 + b4
        dh3, grads['W4'], grads['b4'] = fc_backward(dscores, c['h3'], p['W4'])          # dh3: (N, 64)
        # RELU, then FC1: a3 = h2_flat @ W3 + b3
        da3 = relu_backward(dh3, c['a3'])
        dh2_flat, grads['W3'], grads['b3'] = fc_backward(da3, c['h2_flat'], p['W3'])    # (N, 1024)
        # Undo the flatten: reshape (N, 1024) -> (N, 16, 8, 8)
        dh2 = dh2_flat.reshape(c['h2_shape'])
        # POOL2 -> RELU -> CONV2
        dr2 = pool_backward(dh2, c['r2'], pool_size=2, stride=2)                        # (N, 16, 16, 16)
        da2 = relu_backward(dr2, c['a2'])
        dh1, grads['W2'], grads['b2'] = conv_backward_im2col(da2, c['h1'], p['W2'], stride=1, pad=1,
                                                             cols=c['cols2'])           # dh1: (N, 8, 16, 16)
        # POOL1 -> RELU -> CONV1
        dr1 = pool_backward(dh1, c['r1'], pool_size=2, stride=2)                        # (N, 8, 32, 32)
        da1 = relu_backward(dr1, c['a1'])
        dx, grads['W1'], grads['b1'] = conv_backward_im2col(da1, c['x'], p['W1'], stride=1, pad=1,
                                                            cols=c['cols1'], need_dx=return_dx)
        if return_dx:
            return grads, dx
        return grads

    def update(self, grads, lr, momentum=0.9, weight_decay=0.0):
        """
        [PATCH 2026-10-18] One SGD step with momentum and optional weight decay.

        CS231n neural-networks-3 ("Momentum update"):
            v = mu * v - learning_rate * dx   # integrate velocity
            x += v                            # integrate position
        with v starting at zero; momentum=0 gives the vanilla update
        x += -learning_rate * dx.

        weight_decay is CS231n's L2 regularization (neural-networks-2): adding
        0.5 * weight_decay * sum(W**2) to the loss adds weight_decay * W to the
        gradient, so "every weight is decayed linearly ... towards zero". It is
        applied to the weights W1..W4 only, not to the biases ("it is not common
        to regularize the bias parameters"). For a constant learning rate this
        is the same update as torch.optim.SGD(momentum=mu, weight_decay=wd)
        applied to the weights (checked in validation/day28_backprop_check.py).

        Args:
            grads: dict like self.params (from backward)
            lr: learning rate
            momentum: mu (0 = plain SGD)
            weight_decay: L2 strength lambda (0 = none)
        """
        for name, param in self.params.items():
            grad = grads[name]
            if weight_decay and name.startswith('W'):
                grad = grad + weight_decay * param                # gradient of 0.5 * wd * sum(W**2) is wd * W
            v = momentum * self.velocity.get(name, 0.0) - lr * grad  # .get(..., 0.0): v starts at zero
            param += v                                            # in place, so self.params[name] itself changes
            self.velocity[name] = v

    def predict(self, x):
        """Return predicted class labels."""
        scores = self.forward(x)
        return np.argmax(scores, axis=1)

    def count_parameters(self):
        """Count total learnable parameters."""
        total = 0
        for name, param in self.params.items():
            total += param.size
        return total


# ---------------------------------------------------------------------------
# Utility: Parameter Counter for Arbitrary Architectures
# ---------------------------------------------------------------------------

def count_conv_params(layers):
    """
    Count parameters for a VGGNet-style architecture.

    Args:
        layers: List of tuples describing the architecture.
                Each tuple: ('conv', in_depth, num_filters, filter_size)
                        or: ('pool',)
                        or: ('fc', in_size, out_size)

    Returns:
        total_params: Total parameter count
        per_layer: List of (layer_desc, param_count) tuples
    """
    per_layer = []
    total = 0

    for layer in layers:
        if layer[0] == 'conv':
            _, d_in, k, f = layer
            # Params: (F * F * D_in + 1) * K  (weights + biases)
            params = (f * f * d_in + 1) * k
            per_layer.append((f"CONV{f}-{k}", params))
            total += params
        elif layer[0] == 'pool':
            per_layer.append(("POOL", 0))
        elif layer[0] == 'fc':
            _, d_in, d_out = layer
            params = (d_in + 1) * d_out
            per_layer.append((f"FC-{d_out}", params))
            total += params

    return total, per_layer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Run a quick demonstration of all CNN components."""
    print("=" * 60)
    print("Day 28: CNN Implementation Demo")
    print("=" * 60)

    # 1. Convolution demo
    print("\n--- Convolutional Layer ---")
    x = np.random.randn(1, 3, 8, 8)  # 1 image, 3 channels, 8x8
    w = np.random.randn(4, 3, 3, 3)  # 4 filters, 3x3
    b = np.zeros(4)

    out_naive = conv_forward_naive(x, w, b, stride=1, pad=1)
    out_im2col = conv_forward_im2col(x, w, b, stride=1, pad=1)

    print(f"Input shape:          {x.shape}")
    print(f"Filter shape:         {w.shape}")
    print(f"Output shape (naive): {out_naive.shape}")
    print(f"Output shape (im2col):{out_im2col.shape}")
    print(f"Max difference:       {np.max(np.abs(out_naive - out_im2col)):.2e}")
    print(f"  (should be ~0, confirming both methods are equivalent)")

    # Expected output size: (8 - 3 + 2*1)/1 + 1 = 8
    expected_h = (8 - 3 + 2 * 1) // 1 + 1
    print(f"Expected output H/W:  {expected_h} (matches: {out_naive.shape[2] == expected_h})")

    # 2. Pooling demo
    print("\n--- Pooling Layer ---")
    pool_in = np.random.randn(1, 4, 8, 8)
    pool_out = pool_forward(pool_in, pool_size=2, stride=2)
    print(f"Input shape:  {pool_in.shape}")
    print(f"Output shape: {pool_out.shape}")
    print(f"  (2x2 pooling with stride 2 halves spatial dimensions)")

    # 3. Full network demo
    print("\n--- SimpleCNN Forward Pass ---")
    model = SimpleCNN(num_classes=10)
    images = np.random.randn(2, 3, 32, 32)  # 2 CIFAR-10-sized images
    scores = model.forward(images)
    print(f"Input shape:  {images.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Predictions:  {model.predict(images)}")
    print(f"Total params: {model.count_parameters():,}")

    # [PATCH 2026-10-18] Backward pass demo
    print("\n--- Backward Pass (backprop) ---")
    x = np.random.randn(2, 3, 7, 7)
    w = np.random.randn(4, 3, 3, 3)
    b = np.zeros(4)
    dout = np.random.randn(2, 4, 4, 4)    # stride 2, pad 1: (7 - 3 + 2*1) // 2 + 1 = 4
    dx_naive, dw_naive, db_naive = conv_backward_naive(dout, x, w, stride=2, pad=1)
    dx_fast, dw_fast, db_fast = conv_backward_im2col(dout, x, w, stride=2, pad=1)
    print(f"conv_backward naive vs im2col: max |dx diff| = {np.max(np.abs(dx_naive - dx_fast)):.1e}, "
          f"max |dw diff| = {np.max(np.abs(dw_naive - dw_fast)):.1e}")
    # Gradient check of one filter weight with the centred difference (CS231n neural-networks-3).
    # With L = sum(out * dout), dL/dout = dout, so dL/dw is exactly the dw computed above.
    k, h = (1, 2, 0, 1), 1e-5
    w_plus, w_minus = w.copy(), w.copy()
    w_plus[k] += h
    w_minus[k] -= h
    numeric = (np.sum(conv_forward_im2col(x, w_plus, b, 2, 1) * dout)
               - np.sum(conv_forward_im2col(x, w_minus, b, 2, 1) * dout)) / (2 * h)
    print(f"dw{list(k)}: analytic {dw_fast[k]:.8f}, numerical {numeric:.8f}")

    # CS231n's sanity check: a correct backward pass lets the model memorize a tiny batch.
    print("SimpleCNN, 25 SGD+momentum steps on one batch of 8 random images:")
    model = SimpleCNN(num_classes=10)
    xb, yb = np.random.randn(8, 3, 32, 32), np.arange(8)
    for step in range(25):
        probs = softmax(model.forward(xb))
        loss = cross_entropy_loss(probs, yb)
        grads = model.backward(softmax_cross_entropy_backward(probs, yb))
        model.update(grads, lr=0.01, momentum=0.9)
        if step % 8 == 0 or step == 24:
            print(f"  step {step:2d}: loss {loss:.4f}")

    # 4. Parameter counting demo (VGGNet-16 first few layers)
    print("\n--- VGGNet-16 Parameter Count (first block) ---")
    vgg_layers = [
        ('conv', 3, 64, 3),      # CONV3-64
        ('conv', 64, 64, 3),     # CONV3-64
        ('pool',),               # POOL
        ('conv', 64, 128, 3),    # CONV3-128
        ('conv', 128, 128, 3),   # CONV3-128
        ('pool',),               # POOL
        ('fc', 7*7*512, 4096),   # FC-4096 (simplified)
    ]
    total, per_layer = count_conv_params(vgg_layers)
    for desc, count in per_layer:
        print(f"  {desc:15s} {count:>12,} params")
    print(f"  {'TOTAL':15s} {total:>12,} params")

    print("\n" + "=" * 60)
    print("All components working. Run exercises/ to practice building these.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Implementation Demo")
    parser.add_argument('--demo', action='store_true', default=True,
                        help='Run the demonstration')
    args = parser.parse_args()
    demo()
