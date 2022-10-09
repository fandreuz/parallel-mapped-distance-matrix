import numpy as np


def compute_pts_bin_coords(pts, bin_physical_size):
    assert bin_physical_size.ndim == 1
    assert pts.shape[1] == len(bin_physical_size)
    return np.floor_divide(pts, bin_physical_size).astype(int)


def compute_sort_idxes(pts_bin_coords, bins_per_axis, cut_zeros=True):
    assert bins_per_axis.ndim == 1
    assert pts_bin_coords.shape[1] == len(bins_per_axis)

    linearized_bin_coords = np.ravel_multi_index(
        pts_bin_coords.T, bins_per_axis
    )

    counts = np.bincount(linearized_bin_coords)
    if cut_zeros:
        counts = counts[counts > 0]
    return np.argsort(linearized_bin_coords), counts


def group_buckets(pts, weights, bins_per_axis, bin_physical_size):
    pts_bin_coords = compute_pts_bin_coords(
        pts=pts, bin_physical_size=bin_physical_size
    )
    sort_idxes, pts_count_per_bin = compute_sort_idxes(
        pts_bin_coords=pts_bin_coords,
        bins_per_axis=bins_per_axis,
    )
    pts[:] = pts[sort_idxes]

    if weights is not None:
        weights[:] = weights[sort_idxes]

    first_bin_members = np.concatenate(
        ([0], np.cumsum(pts_count_per_bin)[:-1])
    )
    indexing = sort_idxes[first_bin_members]
    return pts_bin_coords[indexing], pts_count_per_bin
