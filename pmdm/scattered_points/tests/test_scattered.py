import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, Future

from pmdm.scattered_points.core import mapped_distance_matrix
from pmdm.common.test.test_common import FakeExecutor
from pmdm.common.uniform_grid import UniformGrid

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


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_middle_complexity(pool_init):
    pts1 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    pts2 = np.array(
        [[0.5, 0.5], [1, 0.5], [1, 1], [2, 2], [1.5, 1.5]], dtype=float
    )

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.2, 0.2]),
            uniform_grid_size=np.array([16, 16]),
            bins_size=np.array([2, 2]),
            max_distance=2,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(4)

    expected[0] = np.sqrt(0.5) + np.sqrt(1.25) + np.sqrt(2)
    expected[1] = np.sqrt(0.5) + np.sqrt(0.25) + np.sqrt(2) + np.sqrt(0.5)
    expected[2] = np.sqrt(1 + 1.5**2) + np.sqrt(2) + np.sqrt(0.5)
    expected[3] = np.sqrt(2)

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_exact_max_distance_true_boundary(pool_init):
    pts1 = np.array([[0.0, 0.0]], dtype=float)
    pts2 = np.array([[0.3, 0.0], [0.31, 0.0]], dtype=float)

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.1, 0.1]),
            uniform_grid_size=np.array([6, 6]),
            bins_size=np.array([2, 2]),
            max_distance=0.301,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(1)

    expected[0] = 0.3

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_exact_max_distance_false_boundary(pool_init):
    pts1 = np.array([[0.0, 0.0]], dtype=float)
    pts2 = np.array([[0.3, 0.0], [0.31, 0.0]], dtype=float)

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.1, 0.1]),
            uniform_grid_size=np.array([6, 6]),
            bins_size=np.array([2, 2]),
            max_distance=0.301,
            exact_max_distance=False,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(1)

    expected[0] = 0.3 + 0.31

    np.testing.assert_allclose(m, expected)


@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_with_uniform_grid(pool_init):
    ug = UniformGrid(np.array([0.1, 0.1]), np.array([10, 10]))
    pts1 = ug.grid.reshape(-1, 2)
    pts2 = np.array([[0.1, 0.15]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.1, 0.1]),
            uniform_grid_size=np.array([10, 10]),
            bins_size=np.array([2, 2]),
            max_distance=0.051,
            exact_max_distance=True,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(100)

    expected[11] = 0.05
    expected[12] = 0.05

    np.testing.assert_allclose(m, expected)

@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_with_uniform_grid_2(pool_init):
    ug = UniformGrid(np.array([0.1, 0.1]), np.array([10, 10]))
    pts1 = ug.grid.reshape(-1, 2)
    pts2 = np.array([[0.15, 0.15]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.1, 0.1]),
            uniform_grid_size=np.array([10, 10]),
            bins_size=np.array([2, 2]),
            max_distance=0.071,
            exact_max_distance=True,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(100)

    expected[11] = np.sqrt(0.05**2*2)
    expected[12] = expected[11]
    expected[21] = expected[11]
    expected[22] = expected[11]

    np.testing.assert_allclose(m, expected)

@pytest.mark.parametrize("pool_init", thread_pool_initializers)
def test_with_uniform_grid_boundary(pool_init):
    ug = UniformGrid(np.array([0.1, 0.1]), np.array([10, 10]))
    pts1 = ug.grid.reshape(-1, 2)
    pts2 = np.array([[0.0, 0.09]])

    with pool_init() as executor:
        m = mapped_distance_matrix(
            pts1=pts1,
            pts2=pts2,
            uniform_grid_cell_step=np.array([0.1, 0.1]),
            uniform_grid_size=np.array([10, 10]),
            bins_size=np.array([2, 2]),
            max_distance=0.11,
            exact_max_distance=True,
            func=identity,
            executor=executor,
        )

    expected = np.zeros(100)

    expected[0] = 0.09
    expected[1] = 0.01
    expected[11] = np.sqrt(0.1**2 + 0.01**2)

    np.testing.assert_allclose(m, expected)
