import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    # print(f"bias shape : {bias.shape}")

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    print("Dimensions are: \n " + str(out_channels) + " C_out, " + str(out_width) + " out_width\n")
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # initialize tiling dimensions
    c_out_par_dim = nl.tile_size.gemm_stationary_fmax       #128
    num_c_out_tiles = out_channels // c_out_par_dim

    c_in_par_dim = nl.tile_size.pmax        # 128
    num_c_in_tiles = in_channels // c_in_par_dim

    rows_per_chunk = pool_size
    chunks_in_image = out_pool_height

    # process the images in batches
    for b in nl.affine_range(batch_size):
        for c_out_ind in nl.affine_range(num_c_out_tiles):
            bias_temp = nl.load(
                bias[c_out_ind * c_out_par_dim : (c_out_ind + 1) * c_out_par_dim], 
                dtype=bias.dtype,
            )
            for row_chunk in nl.affine_range(chunks_in_image):
                pool_sbuf = nl.ndarray(
                    shape=(c_out_par_dim, out_width, rows_per_chunk),
                    dtype=X_out.dtype,
                    buffer=nl.sbuf,
                )
                for out_row in nl.affine_range(rows_per_chunk):
                    res_psum = nl.zeros((c_out_par_dim, out_width * 1), nl.float32, buffer=nl.psum)
                    for c_in_ind in nl.affine_range(num_c_in_tiles):
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):

                                ## grab input
                                input_mul = nl.ndarray(
                                    shape=(c_in_par_dim, out_width),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf,
                                )
                                nisa.dma_copy(dst=input_mul, src=X[b, c_in_ind * c_in_par_dim:(c_in_ind+1) * c_in_par_dim, row_chunk*rows_per_chunk + out_row+i, j:j+out_width])

                                ## grab filter weight
                                filter_mul = nl.ndarray(
                                    shape=(c_out_par_dim, c_in_par_dim),
                                    dtype=W.dtype,
                                    buffer=nl.sbuf,
                                )
                                filter_mul_T = nl.ndarray(
                                    shape=(c_in_par_dim, c_out_par_dim),
                                    dtype=W.dtype,
                                    buffer=nl.sbuf,
                                )

                                nisa.dma_copy(dst=filter_mul, src=W[c_out_ind * c_out_par_dim : (c_out_ind + 1) * c_out_par_dim, c_in_ind * c_in_par_dim : (c_in_ind + 1) * c_in_par_dim, i, j])
                                filter_psum = nisa.nc_transpose(filter_mul)
                                filter_mul_T = nisa.tensor_copy(filter_psum, engine=nisa.vector_engine)

                                res_psum += nisa.nc_matmul(filter_mul_T, input_mul)
                    
                    res_sbuf = nl.copy(res_psum, dtype=X_out.dtype)
                    res_sbuf[...] = nisa.tensor_tensor(res_sbuf, bias_temp, op=nl.add)
                    pool_sbuf[:, :, out_row] = nisa.tensor_copy(res_sbuf, engine=nisa.vector_engine)

                # perform pooling if necessary:
                
                # reshape our pool_sbuf so that we can compute the max along the correct dimensions
                
                pool_sbuf = pool_sbuf.reshape((c_out_par_dim, out_pool_width, pool_size, rows_per_chunk))
                pool_sbuf = pool_sbuf.reshape((c_out_par_dim, out_pool_width, rows_per_chunk * pool_size))
                pool_sbuf_copy = nisa.tensor_reduce(nl.max, pool_sbuf, axis=[2])
                nisa.dma_copy(dst=X_out[b, c_out_ind * c_out_par_dim : (c_out_ind + 1) * c_out_par_dim, row_chunk, :], src=pool_sbuf_copy)
    return X_out
