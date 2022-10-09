import numpy as np
import numba as nb


def worker(
    pts1,
    pts1_bin_coords,
    pts1_count_per_bin,
    pts2,
    pts2_bin_coords,
    pts2_count_per_bin,
    weights,
    max_distance,
    function,
    exact_max_distance,
):
    pass


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def axis_neighborhood(coord, max_distance_in_cells, nbins, periodic):
    x = np.arange(
        coord - max_distance_in_cells, coord + max_distance_in_cells + 1
    )
    if periodic:
        x = np.mod(x, nbins)
    else:
        x = np.clip(x, a_min=0, a_max=nbins)
    return np.unique(x)


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def get_neighborhood_2d(
    bin_coords, max_distance_in_cells, bins_per_axis, periodic=True
):
    assert bin_coords.ndim == 1 and len(bin_coords) == 2
    rows = axis_neighborhood(
        bin_coords[0], max_distance_in_cells, bins_per_axis[0], periodic
    )
    cols = axis_neighborhood(
        bin_coords[1], max_distance_in_cells, bins_per_axis[1], periodic
    )
    return (cols[None] + rows[:, None] * bins_per_axis[1]).flatten()


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def get_neighborhood_3d(
    bin_coords, max_distance_in_cells, bins_per_axis, periodic=True
):
    assert bin_coords.ndim == 1 and len(bin_coords) == 3
    rows = axis_neighborhood(
        bin_coords[0], max_distance_in_cells, bins_per_axis[0], periodic
    )
    cols = axis_neighborhood(
        bin_coords[1], max_distance_in_cells, bins_per_axis[1], periodic
    )
    depth = axis_neighborhood(
        bin_coords[2], max_distance_in_cells, bins_per_axis[2], periodic
    )
    return (
        depth[None]
        + cols[:, None] * bins_per_axis[1]
        + rows[:, :, None] * bins_per_axis[1] * bins_per_axis[2]
    ).flatten()
