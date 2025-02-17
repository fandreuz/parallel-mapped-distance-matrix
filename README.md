# Parallel MDM

## Mapped distance matrix
The Mapped Distance Matrix (MDM) of two sets $\mathcal{X}, \mathcal{Y}$ of
n-dimensional points is an algebraic structure which is defined in general as
follows, given a mapping $f$:

$$\mathbf{M}(\mathcal{X}, \mathcal{Y}, f)\_{i,j} := f(\Vert \mathcal{X}\_i - \mathcal{Y}\_j\Vert)$$

where $\Vert \cdot \Vert$ is an appropriate distance notion on the space of
definition of $\mathcal{X}$ and $\mathcal{Y}$.

The problem might be augmented by weighting the contributions with a matrix
of weights $\mathbf{W}$; the updated definition is then:

$$\mathbf{M}(\mathcal{X}, \mathcal{Y}, f)\_{i,j} := \mathbf{W}_{i,j} f(\Vert \mathcal{X}\_{i} - \mathcal{Y}\_{j}\Vert)$$

A particularly popular form of the problem (which is also what we treat in this
repository) occurs when weights are defined individually for the members of
$\mathcal{Y}$ (i.e. the columns of $\mathbf{W}$ are taken constants):

$$\mathbf{M}(\mathcal{X}, \mathcal{Y}, f)\_{i,j} := \mathbf{W}\_{j} f(\Vert \mathcal{X}\_{i} - \mathcal{Y}\_{j}\Vert)$$

### A notable case: uniform grid

In general $\mathcal{X}, \mathcal{Y}$ identify two general sets of points. A
few applications allow more assumptions on the two sets. For instance,
$\mathcal{X}$ might be taken to be an uniform grid. In this case a few
interesting optimizization can be taken into account for the computation of the
matrix.

### More assumptions

Practical applications usually require huge sets of points, which causes
memory errors on commonly used devices. This is why it's preferrable to
compute the vector $\tilde{\mathbf{M}}$ defined below instead of $\mathbf{M}$:

$$\tilde{\mathbf{M}}\_{i} := \sum\_{j} \mathbf{M}\_{i,j}$$

For most use cases this is enough.

## Roadmap
- Algorithms
  - [x] Uniform grid algorithm
  - [x] Scattered points algorithm
  - [ ] Fourier-transfor based algorithm
- [ ] Backends
  - [ ] NumPy/Numba
  - [ ] PyTorch
  - [ ] JAX(?)
- [ ] Parallelization
  - [x] Multithreading/Multiprocessing
  - [ ] GPU w/ PyTorch
  - [ ] GPU w/ JAX
  - [ ] CUDA kernels(?)
- [ ] Tests
- [ ] Documentation
- [ ] Benchmark (+comparison with competitors)
  - [ ] CPU
  - [ ] GPU
  - [ ] Several different bin sizes
  - [ ] `pts_per_future`
- [ ] Future
  - [ ] Periodicity
  - [ ] More general about distance definitions
