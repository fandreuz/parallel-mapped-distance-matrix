import numpy as np
import numba as nb
import csr


def worker(
    pts,
    bin_coords,
    weights,
    uniform_grid_cell_step,
    bins_size,
    max_distance,
    function,
    reference_bin,
    exact_max_distance,
    global_matrix_shape,
):
    # location of the lower left point of the non-padded bin in terms
    # of uniform grid cells
    bin_virtual_lower_left = bin_coords * bins_size
    distances = compute_distances(
        pts=pts,
        grid=reference_bin,
        offset=bin_virtual_lower_left * uniform_grid_cell_step,
    )

    if exact_max_distance:
        return compute_mapped_distance_on_subgroup_exact_distance(
            distances=distances,
            weights=weights,
            bin_virtual_lower_left=bin_virtual_lower_left,
            max_distance=max_distance,
            function=function,
            global_matrix_shape=global_matrix_shape,
        ).to_scipy()
    else:
        return compute_mapped_distance_on_subgroup_nexact_distance(
            distances=distances,
            weights=weights,
            bin_virtual_lower_left=bin_virtual_lower_left,
            function=function,
            global_matrix_shape=global_matrix_shape,
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


@nb.jit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_mapped_distance_on_subgroup_nexact_distance(
    distances,
    weights,
    bin_virtual_lower_left,
    function,
    global_matrix_shape,
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
