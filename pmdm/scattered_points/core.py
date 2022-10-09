import numpy as np

from pmdm.common.bucket_utils import group_buckets
from pmdm.common.parallel import distribute_and_start_subproblems


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
    pts_per_future=float("inf"),
    periodic=False,
):
    if weights is None:
        weights = np.ones(len(non_uniform_points), dtype=int)
    dtype = non_uniform_points.dtype

    bins_per_axis = uniform_grid_size // bins_size
    bin_physical_size = uniform_grid_cell_step * bins_size

    pts1_bin_coords, pts1_count_per_bin = group_buckets(
        pts1, None, bins_per_axis, bin_physical_size
    )
    pts2_bin_coords, pts2_count_per_bin = group_buckets(
        pts2, weights, bins_per_axis, bin_physical_size
    )

    max_distance_in_cells = np.ceil(
        max_distance / uniform_grid_cell_step
    ).astype(int)

    trigger = lambda bin_idx, start, size: executor.submit(
        worker,
        pts1,
        pts1_bin_coords,
        pts1_count_per_bin,
        pts2[start : start + size],
        pts2_bin_coords[bin_idx],
        weights[start : start + size],
        max_distance,
        func,
        exact_max_distance,
    )
    futures = distribute_and_start_subproblems(
        trigger,
        nup_count_per_bin=pts2_count_per_bin,
        pts_per_future=pts_per_future,
        executor=executor,
    )
