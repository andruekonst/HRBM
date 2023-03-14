import numpy as np
import numba
import sys


def _pack_to_uint32(bool_array: np.ndarray):
    """
    Args:
        bool_array: Boolean array of shape (n_points, n_values).
    """
    assert sys.byteorder == 'little'
    assert bool_array.ndim == 2
    byte_array = np.packbits(bool_array, axis=1, bitorder='little')
    if byte_array.ndim == 1 or byte_array.shape[1] == 1:
        return byte_array.squeeze(1).astype(np.uint32)
    if byte_array.shape[1] == 2:
        return byte_array.reshape((-1,)).view(np.uint16).astype(np.uint32)
    if byte_array.shape[1] == 3:
        # pad with zero byte
        byte_array = np.concatenate((byte_array, np.zeros_like(byte_array[:, :1])), axis=1)
    if byte_array.shape[1] == 4:
        return byte_array.reshape((-1,)).view(np.uint32).astype(np.uint32)
    raise ValueError(
        f'Too large boolean array second dimension: {bool_array.shape[1]}, '
        f'the packed representation is too long for uint32 ({byte_array.shape[1]} bytes).'
    )


@numba.njit
def _count_ones(n):
    """Count ones in number binary representation (`popcount`).
    Implementation source: https://stackoverflow.com/a/9830282

    Args:
        n: Input number.

    Returns:
        Number of ones in the `n` binary representation.
    """
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32) # for uint64
    return n
