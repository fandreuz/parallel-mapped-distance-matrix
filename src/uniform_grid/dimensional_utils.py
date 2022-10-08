import numpy as np


def extract_slice(arr, lower_bounds, upper_bounds):
    r"""
    Extract a multi-dimensional slice from `arr`. No sanity check of parameters
    is performed, everything is done using NumPy indexing.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array which contains slices that we want to extract.
    lower_bounds: iterable
        An iterable which contains the lower bounds of the slices (included)
        for each dimension (`len(lower_bounds) == arr.ndims`).
    upper_bounds: iterable
        An iterable which contains the upper bounds of the slices (excluded)
        for each dimension (`len(lower_bounds) == arr.ndims`).

    Returns
    -------
    `np.ndarray`
    Slices extracted from `arr`.
    """

    # 1D
    # return arr[lower_bounds[0] : upper_bounds[0]]

    # 2D
    return arr[
        lower_bounds[0] : upper_bounds[0],
        lower_bounds[1] : upper_bounds[1],
    ]

    # 3D
    # return arr[
    #    lower_bounds[0] : upper_bounds[0],
    #    lower_bounds[1] : upper_bounds[1],
    #    lower_bounds[2] : upper_bounds[2],
    # ]

    # ND
    # return arr[
    #    tuple(slice(l, u) for l, u in zip(lower_bounds, upper_bounds))
    # ]


def add_to_slice(arr, val, lower_bounds, upper_bounds):
    r"""
    Add the given NumPy array `val` to a slice of `arr` defined by
    `lower_bounds` and `upper_bounds`. `arr` is modified in place.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array which contains slices that we want to sum to.
    val: np.ndarray
        The NumPy array to be summed to the slice.
    lower_bounds: iterable
        An iterable which contains the lower bounds of the slices (included)
        for each dimension (`len(lower_bounds) == arr.ndims`).
    upper_bounds: iterable
        An iterable which contains the upper bounds of the slices (excluded)
        for each dimension (`len(lower_bounds) == arr.ndims`).
    """

    # 1D
    # arr[lower_bounds[0] : upper_bounds[0]] += val

    # 2D
    arr[
        lower_bounds[0] : upper_bounds[0],
        lower_bounds[1] : upper_bounds[1],
    ] += val

    # 3D
    # arr[
    #    lower_bounds[0] : upper_bounds[0],
    #    lower_bounds[1] : upper_bounds[1],
    #    lower_bounds[2] : upper_bounds[2],
    # ] += val

    # ND
    # arr[
    #    tuple(slice(l, u) for l, u in zip(lower_bounds, upper_bounds))
    # ] += val


def periodic_inner_sum(arr, core_lower_bound, core_upper_bound):
    r"""
    Given a NumPy array `arr` which contains some external slices which should
    be "flipped" and summed periodically, perform the flip operation and return
    the core of `arr`.

    For instance, let's say that the first and last two rows, and the first
    and last three columns in `arr` are periodic parts of `arr`. This means
    that, for instance, `arr[[-1,-2], [-1,-2,-3]]` should be summed to
    `arr[[2,3], [3,4,5]]` in this order.

    The core is the internal, non periodic part. The periodic part is summed
    against the core.

    Parameters
    ----------
    arr: np.ndarray
        A NumPy array which contains periodic slices.
    core_lower_bound: iterable
        Lower bound (included) of the core of `arr`, for each dimension of
        `arr`.
    core_upper_bound: iterable
        Upper bound (not included) of the core of `arr`, for each dimension of
        `arr`.
    """

    arr_size = np.array(arr.shape)

    arr[
        (core_upper_bound[0] - core_lower_bound[0]) : core_upper_bound[0]
    ] += arr[: core_lower_bound[0]]
    arr[
        core_lower_bound[0] : core_lower_bound[0]
        + (arr_size[0] - core_upper_bound[0])
    ] += arr[core_upper_bound[0] :]

    core_rows = slice(core_lower_bound[0], core_upper_bound[0])
    arr = arr[core_rows]

    arr[
        :,
        (core_upper_bound[1] - core_lower_bound[1]) : core_upper_bound[1],
    ] += arr[:, : core_lower_bound[1]]
    arr[
        :,
        core_lower_bound[1] : (
            core_lower_bound[1] + (arr_size[1] - core_upper_bound[1])
        ),
    ] += arr[:, core_upper_bound[1] :]

    return arr[:, core_lower_bound[1] : core_upper_bound[1]]


def generate_uniform_grid(grid_step, grid_size):
    r"""
    Generate an uniform grid according to the given features in `D` dimensions.

    Parameters
    ----------
    grid_step: np.ndarray
        Size of the step of the grid in each direction. Expected a 1D NumPy
        array whose size is `D`.
    grid_size: np.ndarray
        Number of cells in the grid in each direction. Expected a 1D NumPy
        array whose size is `D`.

    Returns
    -------
    `np.ndarray`
    """
    # the tranpose is needed because we want the components in the rightmost
    # part of the shape
    integer_grid = extract_slice(
        np.mgrid, np.zeros_like(grid_size), grid_size
    ).T
    # TODO 3d
    grid = integer_grid * grid_step[None, None]

    return np.swapaxes(grid, 0, 1)


def group_by(a):
    r"""
    Groups the values in `a[:,1]` according to the values in `a[:,0]`. Produces
    a list of lists, each inner list is the set of values in `a[:,1]` that
    share a common value in the first column.

    Parameters
    ----------
    a: np.ndarray
        2D NumPy array, should have two columns. The first one contains the
        reference values to be used for the grouping, the second should
        contain the values to be grouped.

    Returns
    -------
    `list`
    A list of grouped values from the second column of `a`, according to the
    first column of `a`.

    Example
    -------
    The expected returned value for:

        >>> bin_coords = [
        ...     [0, 1],
        ...     [1, 2],
        ...     [1, 3],
        ...     [0, 4],
        ...     [2, 5]
        >>> ]

    is `[[1,4], [2,3], [5]]`.
    """

    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])
