import numpy as np


def compute_pts_bin_coords(pts, bin_physical_size):
    assert bin_physical_size.ndim == 1
    assert pts.shape[1] == len(bin_physical_size)
    return np.floor_divide(pts, bin_physical_size).astype(int)


def compute_sort_idxes(pts_bin_coords, bins_per_axis):
    assert bins_per_axis.ndim == 1
    assert pts_bin_coords.shape[1] == len(bins_per_axis)

    linearized_bin_coords = np.ravel_multi_index(
        pts_bin_coords.T, bins_per_axis
    )

    counts = np.bincount(linearized_bin_coords)
    return np.argsort(linearized_bin_coords), counts[counts > 0]
