import numpy as np
import numba as nb
from concurrent.futures import as_completed
from multiprocessing import Queue
import csr
from scipy.sparse import csr_array

from numpy_dimensional_utils import (
    extract_slice,
    periodic_inner_sum,
    generate_uniform_grid,
    group_by,
)


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


def worker(
    bin_content,
    pts,
    bins_coords,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    return compute_mapped_distance_on_subgroup(
        subgroup_content=pts[bin_content],
        bin_coords=bins_coords[bin_content[0]],
        nup_idxes=bin_content,
        weights=weights[bin_content],
        uniform_grid_cell_step=uniform_grid_cell_step,
        bins_size=bins_size,
        max_distance=max_distance,
        max_distance_in_cells=max_distance_in_cells,
        function=function,
        reference_bin=reference_bin,
        exact_max_distance=exact_max_distance,
        global_matrix_shape=global_matrix_shape,
    ).to_scipy()


def distribute_and_start_subproblems(
    uniform_grid_cell_step,
    uniform_grid_size,
    bins_size,
    pts,
    weights,
    pts_per_future,
    executor,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    bins_per_axis = uniform_grid_size // bins_size

    # periodicity
    pts = np.mod(pts, (uniform_grid_size * uniform_grid_cell_step)[None])

    pts_bin_coords = np.floor_divide(
        pts, uniform_grid_cell_step * bins_size
    ).astype(int)

    # transform the N-Dimensional bins indexing (N is the number of axes)
    # into a linear one (only one index)
    linearized_bin_coords = np.ravel_multi_index(
        pts_bin_coords.T, bins_per_axis
    )
    # augment the linear indexing with the index of the point before using
    # group by, in order to have an index that we can use to access the
    # pts array
    aug_linearized_bin_coords = np.stack(
        (linearized_bin_coords, np.arange(len(pts))), axis=-1
    )
    indexes_inside_bins = group_by(aug_linearized_bin_coords)

    # we create subproblems for each bin (i.e. we split points in the
    # same bin in order to treat at most pts_per_future points in each Future)
    subproblems = extract_subproblems(indexes_inside_bins, pts_per_future)
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
            max_distance_in_cells,
            function,
            reference_bin,
            exact_max_distance,
            global_matrix_shape,
        )
        for bin_content in subproblems
        for subgroup in bin_content
    )


@nb.generated_jit(nopython=True, nogil=True)
def compute_mapped_distance_on_subgroup(
    subgroup_content,
    bin_coords,
    nup_idxes,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    if exact_max_distance:
        return compute_mapped_distance_on_subgroup_exact_distance
    else:
        return compute_mapped_distance_on_subgroup_nexact_distance


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_distances(
    subgroup_content,
    reference_bin,
    bin_coords,
    bins_size,
    uniform_grid_cell_step,
):
    # location of the lower left point of the non-padded bin in terms
    # of uniform grid cells
    bin_virtual_lower_left = bin_coords * bins_size
    # location of the upper right point of the non-padded bin
    bin_virtual_upper_right = bin_virtual_lower_left + bins_size - 1

    # translate the subgroup in order to locate it nearby the reference bin
    # (reminder: the lower left point of the (non-padded) reference bin is
    # [0,0]).
    subgroup_content -= bin_virtual_lower_left * uniform_grid_cell_step

    x_grid, y_grid, cmponents = reference_bin.shape
    x_strides, y_strides, cmponents_strides = reference_bin.strides
    _reference_bin = np.lib.stride_tricks.as_strided(
        reference_bin,
        shape=(x_grid, y_grid, 1, cmponents),
        strides=(x_strides, y_strides, 0, cmponents_strides),
    )
    _subgroup = np.lib.stride_tricks.as_strided(
        subgroup_content,
        shape=(1, 1, *subgroup_content.shape),
        strides=(0, 0, *subgroup_content.strides),
    )

    return np.sqrt(np.sum(np.power(_reference_bin - _subgroup, 2), axis=3))


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_mapped_distance_on_subgroup_nexact_distance(
    subgroup_content,
    bin_coords,
    nup_idxes,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    distances = compute_distances(
        subgroup_content,
        reference_bin,
        bin_coords,
        bins_size,
        uniform_grid_cell_step,
    )
    mapped_distance = np.zeros_like(distances[:, :, 0])
    L, M, N = distances.shape

    data_size = L * M
    rowptrs = np.zeros(global_matrix_shape[0] + 1, dtype=np.int_)
    colinds = np.empty(data_size, dtype=np.int_)
    data = np.empty(data_size, dtype=distances.dtype)

    bin_virtual_lower_left = bin_coords * bins_size
    rows_start = bin_virtual_lower_left[0]
    cols_start = bin_virtual_lower_left[1]

    for i in range(L):
        offset = i * M
        rowptrs[rows_start + i + 1] = rowptrs[rows_start + i] + M
        for j in range(M):
            colinds[offset + j] = j + cols_start
            mapped_distance = 0
            dst = distances[i, j]
            for k in range(N):
                mapped_distance += function(dst[k]) * weights[k]
            data[offset + j] = mapped_distance
    rowptrs[rows_start + L + 1 :] = rowptrs[rows_start + L]

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
    subgroup_content,
    bin_coords,
    nup_idxes,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    max_distance_in_cells,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    distances = compute_distances(
        subgroup_content,
        reference_bin,
        bin_coords,
        bins_size,
        uniform_grid_cell_step,
    )
    mapped_distance = np.zeros_like(distances[:, :, 0])
    L, M, N = distances.shape

    data_size = L * M
    rowptrs = np.zeros(global_matrix_shape[0] + 1, dtype=np.int_)
    colinds = np.empty(data_size, dtype=np.int_)
    data = np.empty(data_size, dtype=distances.dtype)

    bin_virtual_lower_left = bin_coords * bins_size
    rows_start = bin_virtual_lower_left[0]
    cols_start = bin_virtual_lower_left[1]

    for i in range(L):
        offset = i * M
        rowptrs[rows_start + i + 1] = rowptrs[rows_start + i] + M
        for j in range(M):
            colinds[offset + j] = j + cols_start
            mapped_distance = 0
            dst = distances[i, j]
            for k in range(N):
                if dst[k] < max_distance:
                    mapped_distance += function(dst[k]) * weights[k]
            data[offset + j] = mapped_distance
    rowptrs[rows_start + L + 1 :] = rowptrs[rows_start + L]

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
    exact_max_distance=False,
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
    reference_bin = generate_uniform_grid(
        uniform_grid_cell_step, bins_size + 2 * max_distance_in_cells
    )
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
        max_distance_in_cells=max_distance_in_cells,
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
