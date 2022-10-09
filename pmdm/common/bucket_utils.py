import numpy as np


# TODO cython?
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


def compute_pts_bin_coords(pts, bin_physical_size):
    assert bin_physical_size.ndim == 1
    assert pts.shape[1] == len(bin_physical_size)
    return np.floor_divide(pts, bin_physical_size).astype(int)


def pts_indexes_per_bucket(pts_bin_coords, bins_per_axis):
    assert bins_per_axis.ndim == 1
    assert pts_bin_coords.shape[1] == len(bins_per_axis)
    # transform the N-Dimensional bins indexing (N is the number of axes)
    # into a linear one (only one index)
    linearized_bin_coords = np.ravel_multi_index(
        pts_bin_coords.T, bins_per_axis
    )
    # augment the linear indexing with the index of the point before using
    # group by, in order to have an index that we can use to access the
    # pts array
    aug_linearized_bin_coords = np.stack(
        (linearized_bin_coords, np.arange(len(pts_bin_coords))), axis=-1
    )
    return group_by(aug_linearized_bin_coords)
