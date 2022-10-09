import numpy as np

from .dimensional_utils import extract_slice


class UniformGrid:
    def __init__(self, step, size, lazy=True):
        self._step = step
        self._size = size

        if not lazy:
            self.generate()
        else:
            self._grid = None

    @property
    def step(self):
        return self._step

    @property
    def size(self):
        return self._size

    @property
    def grid(self):
        if self._grid is None:
            self.generate()
        return self._grid

    def generate(self):
        r"""Generate an uniform grid according to the given features in `D`
        dimensions.

        Parameters
        ----------
        grid_step: np.ndarray
            Size of the step of the grid in each direction. Expected a 1D NumPy
            array whose size is `D`.
        grid_size: np.ndarray
            Number of cells in the grid in each direction. Expected a 1D NumPy
            array whose size is `D`.

        Returns
        -------
        `np.ndarray`
        """
        # the tranpose is needed because we want the components in the rightmost
        # part of the shape
        integer_grid = extract_slice(
            np.mgrid, np.zeros_like(self.size), self.size
        ).T
        # TODO 3d
        grid = integer_grid * self.step[None, None]

        self._grid = np.swapaxes(grid, 0, 1)
