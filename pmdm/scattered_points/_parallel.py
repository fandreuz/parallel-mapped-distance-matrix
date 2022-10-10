import numpy as np
import numba as nb


def worker(
    pts1,
    pts1_count_per_bin,
    pts2,
    pts2_bin_coords,
    weights,
    max_distance,
    max_distance_in_cells,
    bins_per_axis,
    periodic,
    function,
    exact_max_distance,
    dtype,
):
    if pts1.shape[1] == 2:
        get_neighborhood = get_neighborhood_2d
    elif pts1.shape[1] == 3:
        get_neighborhood = get_neighborhood_3d
    else:
        raise RuntimeError("Not implemented yet")

    neighbor_bins = get_neighborhood(
        pts2_bin_coords, max_distance_in_cells, bins_per_axis, periodic
    )
    pts1_starts = np.concatenate(([0], np.cumsum(pts1_count_per_bin[:-1])))

    pts1_n = np.sum(pts1_count_per_bin[neighbor_bins])
    aggregated_weighted = np.empty(pts1_n, dtype=dtype)

    compute_mapped_distance(
        pts1=pts1,
        pts1_starts=pts1_starts,
        pts1_count_per_bin=pts1_count_per_bin,
        pts1_bin_idxes=neighbor_bins,
        pts2=pts2,
        weights=weights,
        max_distance=max_distance,
        function=function,
        exact_max_distance=exact_max_distance,
        out=aggregated_weighted,
    )

    return aggregated_weighted, neighbor_bins


@nb.generated_jit(nopython=True, nogil=True)
def compute_mapped_distance(
    pts1,
    pts1_starts,
    pts1_count_per_bin,
    pts1_bin_idxes,
    pts2,
    weights,
    max_distance,
    function,
    exact_max_distance,
    out,
):
    if exact_max_distance:
        return compute_mapped_distance_exact_distance
    return compute_mapped_distance_nexact_distance


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_mapped_distance_nexact_distance(
    pts1,
    pts1_starts,
    pts1_count_per_bin,
    pts1_bin_idxes,
    pts2,
    weights,
    max_distance,
    function,
    exact_max_distance,
    out,
):
    pts2_n = len(pts2)

    curr_start = 0
    for bin_idx in pts1_bin_idxes:
        for pts1_idx in range(pts1_count_per_bin[bin_idx]):
            pt1 = pts1[pts1_starts[bin_idx] + pts1_idx]
            acc = 0
            for pts2_idx in range(pts2_n):
                pt2 = pts2[pts2_idx]
                dst = np.sqrt(np.sum(np.power(pt1 - pt2, 2)))
                w = weights[pts2_idx]
                acc += w * function(dst)
            out[curr_start + pts1_idx] = acc
        curr_start += pts1_count_per_bin[bin_idx]


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_mapped_distance_exact_distance(
    pts1,
    pts1_starts,
    pts1_count_per_bin,
    pts1_bin_idxes,
    pts2,
    weights,
    max_distance,
    function,
    exact_max_distance,
    out,
):
    pts2_n = len(pts2)

    curr_start = 0
    for bin_idx in pts1_bin_idxes:
        for pts1_idx in range(pts1_count_per_bin[bin_idx]):
            pt1 = pts1[pts1_starts[bin_idx] + pts1_idx]
            acc = 0
            for pts2_idx in range(pts2_n):
                pt2 = pts2[pts2_idx]
                dst = np.sqrt(np.sum(np.power(pt1 - pt2, 2)))
                if dst < max_distance:
                    w = weights[pts2_idx]
                    acc += w * function(dst)
            out[curr_start + pts1_idx] = acc
        curr_start += pts1_count_per_bin[bin_idx]


def axis_neighborhood(coord, max_distance_in_cells, nbins, periodic):
    x = np.arange(
        coord - max_distance_in_cells, coord + max_distance_in_cells + 1
    )
    if periodic:
        x = np.mod(x, nbins)
    else:
        x = np.clip(x, a_min=0, a_max=nbins - 1)
    return np.unique(x)


def get_neighborhood_2d(
    bin_coords, max_distance_in_cells, bins_per_axis, periodic=True
):
    assert bin_coords.ndim == 1 and len(bin_coords) == 2
    rows = axis_neighborhood(
        bin_coords[0], max_distance_in_cells[0], bins_per_axis[0], periodic
    )
    cols = axis_neighborhood(
        bin_coords[1], max_distance_in_cells[1], bins_per_axis[1], periodic
    )
    return np.unique((cols[None] + rows[:, None] * bins_per_axis[1]).flatten())


def get_neighborhood_3d(
    bin_coords, max_distance_in_cells, bins_per_axis, periodic=True
):
    assert bin_coords.ndim == 1 and len(bin_coords) == 3
    rows = axis_neighborhood(
        bin_coords[0], max_distance_in_cells[0], bins_per_axis[0], periodic
    )
    cols = axis_neighborhood(
        bin_coords[1], max_distance_in_cells[1], bins_per_axis[1], periodic
    )
    depth = axis_neighborhood(
        bin_coords[2], max_distance_in_cells[2], bins_per_axis[2], periodic
    )
    return np.unique(
        (
            depth[None]
            + cols[:, None] * bins_per_axis[1]
            + rows[:, :, None] * bins_per_axis[1] * bins_per_axis[2]
        ).flatten()
    )
