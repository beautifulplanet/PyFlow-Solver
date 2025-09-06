import numpy as np
from numba import njit

@njit(cache=True)
def central_diff_x(arr, dx):
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(1, arr.shape[1]-1):
            out[i, j] = (arr[i, j+1] - arr[i, j-1]) / (2*dx)
    return out

@njit(cache=True)
def central_diff_y(arr, dy):
    out = np.zeros_like(arr)
    for i in range(1, arr.shape[0]-1):
        for j in range(arr.shape[1]):
            out[i, j] = (arr[i+1, j] - arr[i-1, j]) / (2*dy)
    return out

@njit(cache=True)
def upwind_diff_x(arr, dx):
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(1, arr.shape[1]):
            out[i, j] = (arr[i, j] - arr[i, j-1]) / dx
    return out

@njit(cache=True)
def upwind_diff_y(arr, dy):
    out = np.zeros_like(arr)
    for i in range(1, arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = (arr[i, j] - arr[i-1, j]) / dy
    return out
