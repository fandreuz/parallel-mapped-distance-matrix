from concurrent.futures import as_completed
from numbers import Integral

import numpy as np

from pmdm.common.bucket_utils import group_buckets
from pmdm.common.parallel import distribute_and_start_subproblems
from pmdm.scattered_points._parallel import worker


def compute_neighborhood_shifts(max_distance_in_cells):
    return np.arange(-max_distance_in_cells, max_distance_in_cells + 1)


def mapped_distance_matrix(
    pts1,
    pts2,
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    max_distance,
    func,
    executor,
    weights=None,
    exact_max_distance=True,
    pts_per_future=float("inf"),
    periodic=False,
):
    if weights is None:
        weights = np.ones(len(pts2), dtype=int)
    dtype = pts2.dtype
    assert not issubclass(dtype.type, Integral)

    bins_per_axis = uniform_grid_size // bins_size
    bin_physical_size = uniform_grid_cell_step * bins_size
    _, pts1_count_per_bin = group_buckets(
        pts1, None, bins_per_axis, bin_physical_size
    )
    pts2_bin_coords, pts2_count_per_bin = group_buckets(
        pts2, weights, bins_per_axis, bin_physical_size
    )

    max_distance_in_cells = np.ceil(
        max_distance / uniform_grid_cell_step
    ).astype(int)
    neighs_shifts = [
        np.unique(compute_neighborhood_shifts(cells))
        for cells in max_distance_in_cells
    ]

    trigger = lambda bin_idx, start, size: executor.submit(
        worker,
        pts1,
        pts1_count_per_bin,
        pts2[start : start + size],
        pts2_bin_coords[bin_idx],
        weights[start : start + size],
        max_distance,
        neighs_shifts,
        bins_per_axis,
        periodic,
        func,
        exact_max_distance,
        dtype,
    )
    futures = distribute_and_start_subproblems(
        trigger,
        nup_count_per_bin=pts2_count_per_bin,
        pts_per_future=pts_per_future,
        executor=executor,
    )

    result = np.zeros(len(pts1), dtype=dtype)
    for completed in as_completed(futures):
        pts1_result, idxes = completed.result()
        result[idxes] += pts1_result

    return result
