import numpy as np
import numba as nb


def worker(
    pts1,
    pts1_count_per_bin,
    pts2,
    pts2_bin_coords,
    weights,
    max_distance,
    neighbors_shifts,
    bins_per_axis,
    periodic,  # TODO
    function,
    exact_max_distance,
    dtype,
):
    neighbor_bins = get_neighborhood(
        pts2_bin_coords,
        neighbors_shifts,
        bins_per_axis,
        len(pts1_count_per_bin),
    )
    pts1_starts = np.concatenate(([0], np.cumsum(pts1_count_per_bin[:-1])))

    pts1_n = np.sum(pts1_count_per_bin[neighbor_bins])
    aggregated_weighted = np.empty(pts1_n, dtype=dtype)

    if exact_max_distance:
        compute_mapped_distance_exact_distance(
            pts1=pts1,
            pts1_starts=pts1_starts,
            pts1_count_per_bin=pts1_count_per_bin,
            pts1_bin_idxes=neighbor_bins,
            pts2=pts2,
            weights=weights,
            max_distance=max_distance,
            function=function,
            out=aggregated_weighted,
        )
    else:
        compute_mapped_distance_nexact_distance(
            pts1=pts1,
            pts1_starts=pts1_starts,
            pts1_count_per_bin=pts1_count_per_bin,
            pts1_bin_idxes=neighbor_bins,
            pts2=pts2,
            weights=weights,
            max_distance=max_distance,
            function=function,
            out=aggregated_weighted,
        )

    return aggregated_weighted, neighbor_bins


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


def get_neighborhood_2d(bin_coords, neighs_shift, bins_per_axis):
    rows = bin_coords[0] + neighs_shift[0]
    cols = bin_coords[1] + neighs_shift[1]
    return np.unique((cols[None] + rows[:, None] * bins_per_axis[1]).flatten())


def get_neighborhood_3d(bin_coords, neighs_shift, bins_per_axis):
    rows = bin_coords[0] + neighs_shift[0]
    cols = bin_coords[1] + neighs_shift[1]
    depths = bin_coords[2] + neighs_shift[2]
    return np.unique(
        (
            depth[None]
            + cols[:, None] * bins_per_axis[1]
            + rows[:, :, None] * bins_per_axis[1] * bins_per_axis[2]
        ).flatten()
    )


def get_neighborhood(bin_coords, neighs_shift, bins_per_axis, nbins):
    ndims = len(bin_coords)
    if ndims == 2:
        neighborhood = get_neighborhood_2d(
            bin_coords, neighs_shift, bins_per_axis
        )
    elif ndims == 3:
        neighborhood = get_neighborhood_3d(
            bin_coords, neighs_shift, bins_per_axis
        )
    else:
        raise RuntimeError("Not implemented yet")

    # TODO periodicity

    return np.clip(neighborhood, a_min=0, a_max=nbins - 1)
