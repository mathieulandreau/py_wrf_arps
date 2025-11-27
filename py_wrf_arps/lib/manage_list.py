import numpy as np

def is_iterable(x) :
    try:
        iter(x)
        return True
    except :
        return False
    
def longest_increasing_sequence(arr):
    """
    Description
        Find the longest increasing sequence of a list or array
        Based on : https://reintech.io/blog/python-longest-increasing-subsequence
        The difference with longest_increasing_subsequence is that every element should be in the same slice
    Parameters
        arr : list or 1D numpy.array
    Return 
        maximum : int : the length of the subsequence
        i0 : int : the index of the first element of the subsequence in arr 
        s : slice : slice of the subsequence in arr
    """
    n = len(arr)
    lis = [1]*n
    for i in range (1 , n):
        j = i
        while j<n-1 and arr[j+1] >= arr[j]:
            lis[i] = lis[i]+1
            j=j+1
    maximum = 1
    i0 = 0
    for i in range(len(lis)):
        if lis[i] > maximum :
            maximum = lis[i]
            i0 = i
    s = slice(i0, i0+maximum)
    return maximum, i0, s


def unstag(vec, dim = 0):
    """
    Description
        Unstagger a staggered variable
    Parameters
        vec : np.array : staggered along dimension dim
    Optional
        dim : the dimension along which we want to unstagger, default 0
    Return 
        np.array : unstaggered array
    """
    if dim == 0:
        return 0.5*(vec[1:] + vec[:-1])
    elif dim == 1:
        return 0.5*(vec[:, 1:] + vec[:, :-1])
    elif dim == 2:
        return 0.5*(vec[:, :, 1:] + vec[:, :, :-1])
    elif dim == 3:
        return 0.5*(vec[:, :, :, 1:] + vec[:, :, :, :-1])
    else :
        print("unknown dim, unstag with dim=0")
        return unstag(vec, 0)

def moving_average(a, n, axis=0):
    # source : https://python.plainenglish.io/how-to-avoid-to-drop-the-edge-of-the-data-at-moving-averaging-python-tutorial-5d243ee8efff
    padded = np.apply_along_axis(np.pad, axis, a, (n//2, n-1-n//2), mode='edge')
    kernel = np.ones(n) / n
    return np.apply_along_axis(np.convolve, axis=axis, arr=padded, v=kernel, mode='valid')
    
def moving_average2(a, n, axis=0):
    a2 = moving_average(a, n, axis)
    if axis==0 :
        a2[:n//2] = a2[n//2:n//2+1]
        a2[-n//2:] = a2[-n//2:-n//2+1]
    if axis==1 :
        a2[:,:n//2] = a2[:,n//2:n//2+1]
        a2[:,-n//2:] = a2[:,-n//2:-n//2+1]
    if axis==2 :
        a2[:,:,:n//2] = a2[:,:,n//2:n//2+1]
        a2[:,:,-n//2:] = a2[:,:,-n//2:-n//2+1]
    if axis==3 :
        a2[:,:,:,:n//2] = a2[:,:,:,n//2:n//2+1]
        a2[:,:,:,-n//2:] = a2[:,:,:,-n//2:-n//2+1]
    return a2
        
