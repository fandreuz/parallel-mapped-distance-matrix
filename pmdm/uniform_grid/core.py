from concurrent.futures import as_completed

import numpy as np
import numba as nb
import csr
from scipy.sparse import csr_array

from pmdm.common.bucket_utils import pts_indexes_per_bucket
from pmdm.common.uniform_grid import UniformGrid
from pmdm.common.dimensional_utils import periodic_inner_sum


def extract_subproblems(indexes, n_per_subgroup):
    if n_per_subgroup != -1:
        return map(
            # here we apply the finer granularity (#pts per future)
            lambda arr: np.array_split(
                arr, np.ceil(len(arr) / n_per_subgroup)
            ),
            indexes,
        )
    else:
        # we transform the lists of indexes in indexes_inside_bins to NumPy
        # arrays. we also wrap them into 1-element tuples because of how we
        # treat them in client.map
        return map(lambda arr: (np.array(arr),), indexes)


def distribute_and_start_subproblems(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    pts,
    weights,
    pts_per_future,
    executor,
    max_distance,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    bins_per_axis = uniform_grid_size // bins_size

    # periodicity
    pts = np.mod(pts, (uniform_grid_size * uniform_grid_cell_step)[None])

    bin_physical_size = uniform_grid_cell_step * bins_size
    indexes_inside_buckets, pts_bin_coords = pts_indexes_per_bucket(
        pts=pts,
        bin_physical_size=bin_physical_size,
        bins_per_axis=bins_per_axis,
    )

    # we create subproblems for each bin (i.e. we split points in the
    # same bin in order to treat at most pts_per_future points in each Future)
    subproblems = extract_subproblems(indexes_inside_buckets, pts_per_future)
    # each subproblem is treated by a single Future. each bin spawns one or
    # more subproblems.

    return (
        executor.submit(
            worker,
            subgroup,
            pts,
            pts_bin_coords,
            weights,
            uniform_grid_cell_step,
            bins_size,
            max_distance,
            function,
            reference_bin,
            exact_max_distance,
            global_matrix_shape,
        )
        for bin_content in subproblems
        for subgroup in bin_content
    )


def worker(
    bin_content,
    pts,
    bins_coords,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    bin_coords = bins_coords[bin_content[0]]
    # location of the lower left point of the non-padded bin in terms
    # of uniform grid cells
    bin_virtual_lower_left = bin_coords * bins_size
    distances = compute_distances(
        pts=pts[bin_content],
        grid=reference_bin,
        offset=bin_virtual_lower_left * uniform_grid_cell_step,
    )
    return compute_mapped_distance_on_subgroup(
        distances=distances,
        weights=weights[bin_content],
        bin_virtual_lower_left=bin_virtual_lower_left,
        max_distance=max_distance,
        function=function,
        global_matrix_shape=global_matrix_shape,
        exact_max_distance=exact_max_distance,
    ).to_scipy()


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_distances(
    pts,
    grid,
    offset,
):
    # translate the subgroup in order to locate it nearby the reference bin
    # (reminder: the lower left point of the (non-padded) reference bin is
    # [0,0]).
    pts -= offset

    x_grid, y_grid, cmponents = grid.shape
    x_strides, y_strides, cmponents_strides = grid.strides
    _grid = np.lib.stride_tricks.as_strided(
        grid,
        shape=(x_grid, y_grid, 1, cmponents),
        strides=(x_strides, y_strides, 0, cmponents_strides),
    )
    _pts = np.lib.stride_tricks.as_strided(
        pts,
        shape=(1, 1, *pts.shape),
        strides=(0, 0, *pts.strides),
    )

    return np.sqrt(np.sum(np.power(_grid - _pts, 2), axis=3))


@nb.generated_jit(nopython=True, nogil=True)
def compute_mapped_distance_on_subgroup(
    distances,
    weights,
    bin_virtual_lower_left,
    max_distance,
    function,
    global_matrix_shape,
    exact_max_distance,
):
    if exact_max_distance:
        return compute_mapped_distance_on_subgroup_exact_distance
    return compute_mapped_distance_on_subgroup_nexact_distance


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_mapped_distance_on_subgroup_nexact_distance(
    distances,
    weights,
    bin_virtual_lower_left,
    max_distance,  # unused
    function,
    global_matrix_shape,
    exact_max_distance,  # unused
):
    mapped_distance = np.zeros_like(distances[..., 0])
    nrows, ncols, nnupt = distances.shape

    data_size = nrows * ncols
    rowptrs = np.zeros(global_matrix_shape[0] + 1, dtype=np.int_)
    colinds = np.empty(data_size, dtype=np.int_)
    data = np.empty(data_size, dtype=distances.dtype)

    rows_start = bin_virtual_lower_left[0]
    cols_start = bin_virtual_lower_left[1]

    for i in range(nrows):
        offset = i * ncols
        for j in range(ncols):
            colinds[offset + j] = j + cols_start
            mapped_distance = 0

            dst = distances[i, j]
            for k in range(nnupt):
                mapped_distance += function(dst[k]) * weights[k]
            data[offset + j] = mapped_distance

        rowptrs[rows_start + i + 1] = rowptrs[rows_start + i] + ncols
    rowptrs[rows_start + nrows + 1 :] = rowptrs[rows_start + nrows]

    return csr.create(
        global_matrix_shape[0],
        global_matrix_shape[1],
        data_size,
        rowptrs,
        colinds,
        data,
    )


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_mapped_distance_on_subgroup_exact_distance(
    distances,
    weights,
    bin_virtual_lower_left,
    max_distance,
    function,
    global_matrix_shape,
    exact_max_distance,  # unused
):
    mapped_distance = np.zeros_like(distances[..., 0])
    nrows, ncols, nnupt = distances.shape

    data_size = nrows * ncols
    rowptrs = np.zeros(global_matrix_shape[0] + 1, dtype=np.int_)
    colinds = np.empty(data_size, dtype=np.int_)
    data = np.empty(data_size, dtype=distances.dtype)

    rows_start = bin_virtual_lower_left[0]
    cols_start = bin_virtual_lower_left[1]

    for i in range(nrows):
        offset = i * ncols
        for j in range(ncols):
            colinds[offset + j] = j + cols_start
            mapped_distance = 0

            dst = distances[i, j]
            for k in range(nnupt):
                if dst[k] < max_distance:
                    mapped_distance += function(dst[k]) * weights[k]
            data[offset + j] = mapped_distance

        rowptrs[rows_start + i + 1] = rowptrs[rows_start + i] + ncols
    rowptrs[rows_start + nrows + 1 :] = rowptrs[rows_start + nrows]

    return csr.create(
        global_matrix_shape[0],
        global_matrix_shape[1],
        data_size,
        rowptrs,
        colinds,
        data,
    )


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
    pts_per_future=-1,
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

    # build a reference padded bin. the padding is given by taking twice
    # (in each direction) the value of max_distance. the lower_left point is
    # set to max_distance because we want the (0,0) point to be the first
    # point inside the non-padded bin.
    reference_bin = UniformGrid(
        step=uniform_grid_cell_step,
        size=bins_size + 2 * max_distance_in_cells,
        lazy=False,
    ).grid

    lower_left = -(max_distance_in_cells * uniform_grid_cell_step)
    reference_bin += lower_left + cell_reference_point_offset

    shape = tuple(uniform_grid_size + 2 * max_distance_in_cells)
    # split and distribute subproblems to the workers
    futures = distribute_and_start_subproblems(
        uniform_grid_cell_step=uniform_grid_cell_step,
        uniform_grid_size=uniform_grid_size,
        bins_size=bins_size,
        pts=non_uniform_points,
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
