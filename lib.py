import sys
import typing
from typing import Union, Optional, Callable

import numpy
from pyscf.lib import logger
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gcrotmk

'''
GMRES/GCROT(m,k) for solving Green's function linear equations
'''

OptionalArray = Optional[numpy.ndarray]


def _unpack(v=None, vs=None):
    res = None

    if v is not None and vs is None:
        v = numpy.asarray(v)
        res = v.reshape(1, -1) if v.ndim == 1 else None

    if vs is not None and v is None:
        vs = numpy.asarray(vs)
        res = vs if vs.ndim == 2 else vs.reshape(1, -1) if vs.ndim == 1 else None

    return res


def gmres(h, bs: OptionalArray = None, b: OptionalArray = None,
          xs0: OptionalArray = None, x0: OptionalArray = None,
          diag: OptionalArray = None,
          m: int = 100, tol: float = 1e-6, max_cycle: int = 200,
          verbose=0, stdout: typing.TextIO = sys.stdout) -> numpy.ndarray:
    """Solve a linear system using the GMRES algorithm.

    Solves the linear equation h x = b using the GMRES (Generalized Minimal Residual) algorithm.
    GMRES is an iterative method for solving nonsymmetric linear systems by approximating the solution
    in a Krylov subspace with minimal residual.

    Args:
        h (Union[Callable, numpy.ndarray]):
            Either a callable representing the matrix-vector product operation or a matrix itself.
        bs (ArrayLike):
            Right-hand side vector(s). If multiple vectors are provided, they should be stacked vertically.
        b (ArrayLike):
            Deprecated. Equivalent to providing `bs`.
        xs0 (ArrayLike, optional):
            Initial guess for the solution. Defaults to `None`.
        x0 (ArrayLike, optional):
            Deprecated. Equivalent to providing `xs0`.
        diag (ArrayLike, optional):
            Diagonal preconditioner for the linear operator `h`. Defaults to `None`.
        m (int, optional):
            Number of Krylov basis vectors used in the GMRES algorithm. Defaults to 30.
        tol (float, optional):
            Tolerance for terminating the iteration. Defaults to 1e-6.
        max_cycle (int, optional):
            Maximum number of iterations. Defaults to 200.
        verbose (int, optional):
            Verbosity level. If greater than 0, detailed logs are printed. Defaults to 0.
        stdout (typing.TextIO, optional):
            Output stream for logging. Defaults to `sys.stdout`.

    Returns:
        numpy.ndarray:
            Solution vector(s) of the linear system. If multiple right-hand side vectors are provided,
            the solutions are stacked vertically.

    Raises:
        ValueError:
            If convergence to tolerance is not achieved within the specified number of iterations.

    Note:
        The `b` and `x0` arguments are provided for compatibility.

    """
    log = logger.Logger(stdout, verbose)
    bs = _unpack(b, bs)
    xs0 = _unpack(x0, xs0)

    assert bs is not None
    nb, n = bs.shape
    nnb = nb * n

    assert diag.shape == (n,)

    if callable(h):
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        # noinspection PyArgumentList
        hop = LinearOperator((nnb, nnb), matvec=matvec)
    else:
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h.dot(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        assert h.shape == (n, n)

        # noinspection PyArgumentList
        hop = LinearOperator((nnb, nnb), matvec=matvec)

    mop = None
    if diag is not None:
        diag = diag.reshape(-1)

        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([x / diag for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        # noinspection PyArgumentList
        mop = LinearOperator((nnb, nnb), matvec=matvec)

    num_iter = 0

    def callback(rk):
        nonlocal num_iter
        num_iter += 1
        log.debug(f"GMRES: iter = {num_iter:4d}, residual = {numpy.linalg.norm(rk) / nb:6.4e}")

    log.debug(f"\nGMRES Start")
    log.debug(f"GMRES: nb  = {nb:4d}, n = {n:4d},  m = {m:4d}")
    log.debug(f"GMRES: tol = {tol:4.2e}, max_cycle = {max_cycle:4d}")

    if xs0 is not None:
        xs0 = xs0.reshape(-1)

    # noinspection PyTypeChecker
    xs, info = gcrotmk(
        hop, bs.reshape(-1), x0=xs0, M=mop,
        maxiter=max_cycle, callback=callback, m=m,
        tol=tol / nb, atol=tol / nb
    )

    if info > 0:
        print(f"Convergence to tolerance not achieved in {info} iterations")
        res = hop.matvec(xs) - bs.reshape(-1)
        print(f"Final residual norm = {numpy.linalg.norm(res) / nb:6.4e}")

    if nb == 1:
        xs = xs.reshape(n, )
    else:
        xs = xs.reshape(nb, n)

    return xs
