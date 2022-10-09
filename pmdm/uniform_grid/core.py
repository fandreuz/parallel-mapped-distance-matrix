from concurrent.futures import as_completed

import numpy as np
from scipy.sparse import csr_array

from pmdm.common.bucket_utils import group_buckets
from pmdm.common.dimensional_utils import periodic_inner_sum
from pmdm.common.parallel import distribute_and_start_subproblems
from pmdm.uniform_grid._parallel import worker


def prepare_reference_bin(step, bins_size, max_distance_in_cells):
    from pmdm.common.uniform_grid import UniformGrid

    reference_bin = UniformGrid(
        step=step,
        size=bins_size + 2 * max_distance_in_cells,
        lazy=False,
    ).grid
    lower_left = -(max_distance_in_cells * step)
    return reference_bin + lower_left


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

    bins_per_axis = uniform_grid_size // bins_size
    bin_physical_size = uniform_grid_cell_step * bins_size
    pts_bin_coords, pts_count_per_bin = group_buckets(
        non_uniform_points, weights, bins_per_axis, bin_physical_size
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
    trigger = lambda bin_idx, start, size: executor.submit(
        worker,
        non_uniform_points[start : start + size],
        pts_bin_coords[bin_idx],
        weights[start : start + size],
        uniform_grid_cell_step,
        bins_size,
        max_distance,
        func,
        reference_bin,
        exact_max_distance,
        shape,
    )
    futures = distribute_and_start_subproblems(
        trigger,
        nup_count_per_bin=pts_count_per_bin,
        pts_per_future=pts_per_future,
        executor=executor,
    )

    mapped_distance = csr_array(shape, dtype=dtype)
    for completed in as_completed(futures):
        mapped_distance += completed.result()

    return periodic_inner_sum(
        mapped_distance.todense(),
        max_distance_in_cells,
        uniform_grid_size + max_distance_in_cells,
    )
