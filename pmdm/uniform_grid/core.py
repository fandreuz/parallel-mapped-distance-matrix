from concurrent.futures import as_completed

import numpy as np
from scipy.sparse import csr_array

from pmdm.common.bucket_utils import (
    compute_sort_idxes,
    compute_pts_bin_coords,
)
from pmdm.common.dimensional_utils import periodic_inner_sum
from pmdm.uniform_grid._parallel import worker


def distribute_and_start_subproblems(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    pts,
    pts_bin_coords,
    pts_count_per_bin,
    weights,
    pts_per_future,
    executor,
    max_distance,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    start = lambda bin_idx, start, size: executor.submit(
        worker,
        pts[start : start + size],
        pts_bin_coords[bin_idx],
        weights[start : start + size],
        uniform_grid_cell_step,
        bins_size,
        max_distance,
        function,
        reference_bin,
        exact_max_distance,
        global_matrix_shape,
    )

    curr_start = 0
    futures = []
    for bin_idx, count in enumerate(pts_count_per_bin):
        while count > pts_per_future:
            count -= pts_per_future
            futures.append(start(bin_idx, curr_start, pts_per_future))
            curr_start += pts_per_future
        if count > 0:
            futures.append(start(bin_idx, curr_start, count))
            curr_start += count
    return futures


def prepare_reference_bin(step, bins_size, max_distance_in_cells):
    from pmdm.common.uniform_grid import UniformGrid

    reference_bin = UniformGrid(
        step=step,
        size=bins_size + 2 * max_distance_in_cells,
        lazy=False,
    ).grid
    lower_left = -(max_distance_in_cells * step)
    return reference_bin + lower_left


def group_buckets(
    pts, weights, uniform_grid_cell_step, uniform_grid_size, bins_size
):
    bins_per_axis = uniform_grid_size // bins_size

    bin_physical_size = uniform_grid_cell_step * bins_size
    pts_bin_coords = compute_pts_bin_coords(
        pts=pts, bin_physical_size=bin_physical_size
    )
    sort_idxes, pts_count_per_bin = compute_sort_idxes(
        pts_bin_coords=pts_bin_coords,
        bins_per_axis=bins_per_axis,
    )
    pts[:] = pts[sort_idxes]
    weights[:] = weights[sort_idxes]

    first_bin_members = np.concatenate(
        ([0], np.cumsum(pts_count_per_bin)[:-1])
    )
    indexing = sort_idxes[first_bin_members]
    return pts_bin_coords[indexing], pts_count_per_bin


def mapped_distance_matrix(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    non_uniform_points,
    max_distance,
    func,
    executor,
    weights=None,
    exact_max_distance=True,
    pts_per_future=float("inf"),
    cell_reference_point_offset=0,
):
    if weights is None:
        weights = np.ones(len(non_uniform_points), dtype=int)
    dtype = non_uniform_points.dtype
    if len(non_uniform_points) == 0:
        return np.zeros(uniform_grid_size, dtype=dtype)

    # bins should divide properly the grid
    assert np.all(np.mod(uniform_grid_size, bins_size) == 0)

    max_distance_in_cells = np.ceil(
        max_distance / uniform_grid_cell_step
    ).astype(int)

    # periodicity
    non_uniform_points = np.mod(
        non_uniform_points, (uniform_grid_size * uniform_grid_cell_step)[None]
    )

    pts_bin_coords, pts_count_per_bin = group_buckets(
        non_uniform_points,
        weights,
        uniform_grid_cell_step,
        uniform_grid_size,
        bins_size,
    )
    reference_bin = (
        prepare_reference_bin(
            step=uniform_grid_cell_step,
            bins_size=bins_size,
            max_distance_in_cells=max_distance_in_cells,
        )
        + cell_reference_point_offset
    )

    shape = tuple(uniform_grid_size + 2 * max_distance_in_cells)
    # split and distribute subproblems to the workers
    futures = distribute_and_start_subproblems(
        uniform_grid_cell_step=uniform_grid_cell_step,
        uniform_grid_size=uniform_grid_size,
        bins_size=bins_size,
        pts=non_uniform_points,
        pts_bin_coords=pts_bin_coords,
        pts_count_per_bin=pts_count_per_bin,
        pts_per_future=pts_per_future,
        executor=executor,
        weights=weights,
        max_distance=max_distance,
        reference_bin=reference_bin,
        function=func,
        exact_max_distance=exact_max_distance,
        global_matrix_shape=shape,
    )

    mapped_distance = csr_array(shape, dtype=dtype)
    for completed in as_completed(futures):
        mapped_distance += completed.result()

    return periodic_inner_sum(
        mapped_distance.todense(),
        max_distance_in_cells,
        uniform_grid_size + max_distance_in_cells,
    )
