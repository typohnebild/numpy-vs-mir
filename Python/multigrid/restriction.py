import numpy as np


def restriction(A):
    """
        applies simple restriction to A
        @param A n x n matrix
        @return (n//2 +1, n//2 + 1) matrix
    """
    alpha = len(A.shape)
    ret = np.zeros(np.array(A.shape) // 2 + 1)
    end = ret.shape[0] - (A.shape[0] + 1) % 2

    if alpha == 1:
        ret[:end:] = A[::2]
        ret[-1] = A[-1]
    elif alpha == 2:
        ret[:end:, :end:] = A[::2, ::2]
        ret[:end, -1] = A[::2, -1]
        ret[-1, :end] = A[-1, ::2]
        ret[-1, -1] = A[-1, -1]
    elif alpha == 3:
        ret[:end:, :end:, :end:] = A[::2, ::2, ::2]

        ret[:end, :end, -1] = A[::2, ::2, -1]
        ret[-1, :end, :end] = A[-1, ::2, ::2]
        ret[:end, -1, :end] = A[::2, -1, ::2]

        ret[:end, -1, -1] = A[::2, -1, -1]
        ret[-1, :end, -1] = A[-1, ::2, -1]
        ret[-1, -1, :end] = A[-1, -1, ::2]

        ret[-1, -1, -1] = A[-1, -1, -1]
    else:
        raise ValueError('restriction: invalid dimension')
    return ret


def weighted_restriction(A):
    alpha = len(A.shape)
    B = np.zeros(np.array(A.shape) // 2)

    if alpha == 1:
        # TODO
        return A[1::2]
    if alpha == 2:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = (A[2 * i][2 * j] / 2 +
                           (A[2 * i + 1][2 * j] +
                            A[2 * i - 1][2 * j] +
                            A[2 * i][2 * j + 1] +
                            A[2 * i][2 * j - 1]
                            ) / 4 +
                           (A[2 * i + 1][2 * j + 1] +
                            A[2 * i + 1][2 * j - 1] +
                            A[2 * i - 1][2 * j + 1] +
                            A[2 * i - 1][2 * j - 1]
                            ) / 8)
        return B
    if alpha == 3:
        # TODO
        return A[1::2, 1::2, 1::2]
    else:
        raise ValueError('weighted restriction: invalid dimension')
