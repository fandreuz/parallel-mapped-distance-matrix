import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, Future

from pmdm.scattered_points.core import mapped_distance_matrix
from pmdm.common.test.test_common import FakeExecutor

import pytest


@nb.njit
def identity(x):
    return x


fake_pool_init = lambda: FakeExecutor()
thread_pool2_init = lambda: ThreadPoolExecutor(2)
thread_pool4_init = lambda: ThreadPoolExecutor(4)
thread_pool_initializers = (
    fake_pool_init,
    thread_pool2_init,
    thread_pool4_init,
)

# -------


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_simple(pool_init):
    pts1 = np.array([[0, 0], [1, 1]], dtype=float)
    pts2 = np.array([[0.5, 0.5], [1, 0.5]], dtype=float)

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.6, 0.6]),
            uniform_grid_size=np.array([2, 2]),
            bins_size=np.array([2, 1]),
            max_distance=2,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(2)

    expected[0] = np.sqrt(0.5) + np.sqrt(1.25)
    expected[1] = np.sqrt(0.5) + np.sqrt(0.25)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_simple_reversed_bins(pool_init):
    pts1 = np.array([[0, 0], [1, 1]], dtype=float)
    pts2 = np.array([[0.5, 0.5], [1, 0.5]], dtype=float)

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.6, 0.6]),
            uniform_grid_size=np.array([2, 2]),
            bins_size=np.array([1, 2]),
            max_distance=2,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(2)

    expected[0] = np.sqrt(0.5) + np.sqrt(1.25)
    expected[1] = np.sqrt(0.5) + np.sqrt(0.25)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_simple_restrictive_distance(pool_init):
    pts1 = np.array([[0, 0], [1, 1]], dtype=float)
    pts2 = np.array([[0.5, 0.5], [1, 0.5]], dtype=float)

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.6, 0.6]),
            uniform_grid_size=np.array([2, 2]),
            bins_size=np.array([2, 1]),
            max_distance=1,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(2)

    expected[0] = np.sqrt(0.5)
    expected[1] = np.sqrt(0.5) + np.sqrt(0.25)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_simple_reversed_bins_restrictive_distance(pool_init):
    pts1 = np.array([[0, 0], [1, 1]], dtype=float)
    pts2 = np.array([[0.5, 0.5], [1, 0.5]], dtype=float)

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.6, 0.6]),
            uniform_grid_size=np.array([2, 2]),
            bins_size=np.array([1, 2]),
            max_distance=1,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(2)

    expected[0] = np.sqrt(0.5)
    expected[1] = np.sqrt(0.5) + np.sqrt(0.25)

    np.testing.assert_allclose(m, expected)
