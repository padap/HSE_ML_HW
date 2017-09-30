def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    res = 1
    for c, lst in enumerate(x):
        if c >= len(lst):
            return res
        res *= lst[c] if lst[c] != 0 else 1
    return res


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    return sorted(x) == sorted(y)


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    last = x[0] == 0
    MAX = -666
    for i in x[1:]:
        if i > MAX and last:
            MAX = i
        last = True if i == 0 else False
    return MAX


def convert_image(img, coefs=[0.299, 0.587, 0.114]):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    res = []
    for line in img:
        temp_res = []
        for k in line:
            temp_res.append(sum([a*b for a, b in zip(k, coefs)]))
        res.append(temp_res)
    return res


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    elements, counter = [x[0]], [0]
    for i in x:
        if elements[-1] == i:
            counter[-1] += 1
        else:
            elements.append(i)
            counter.append(1)
    return elements, counter


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    M = []
    import math

    def dist(l1, l2):
        return math.sqrt(sum([(i-k)**2 for i, k in zip(l1, l2)]))

    for i in x:
        temp = []
        for k in y:
            temp.append(dist(i, k))
        M.append(temp)
    return M

