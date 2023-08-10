import sys
from functools import reduce
from typing import Callable, Tuple

import numpy
import scipy

ElectronSpinNumber = Tuple[int, int]
prod: Callable[[Tuple], int] = lambda x: reduce(lambda a, b: a * b, x)

MAX_MEMORY_MB = 8000.0  # MB
MAX_MEMORY_GB = MAX_MEMORY_MB / 1024

import pyscf
from pyscf.fci.addons import des_a

import epcc.fci
from epcc.hol_model import HolModel
from epcc.fci import contract_1e
from epcc.fci import contract_ep_rspace as contract_ep
from epcc.fci import contract_pp

# import .lib
from lib import gmres

def gen_hop(t: numpy.ndarray, g: numpy.ndarray, w: numpy.ndarray,
            nelec: ElectronSpinNumber = (1, 0), nph_max: int = 4) -> Callable:
    r"""
    This function generates the Hamiltonian operator in the Fock space:
        :math:`H c_I |\Psi_I\rangle = s_{IJ} c_J |\Psi_J\rangle`

    Args:
        t: numpy.ndarray (nsite, nsite)
            hopping matrix
        g: numpy.ndarray (nmode, nsite, nsite)
            electron-phonon coupling matrix
        w: numpy.ndarray (nmode, )
            phonon frequency matrix
        nelec: ElectronSpinNumber
            number of electrons
        nph_max: int
            maximum number of phonons

    Returns:
        hop: Callable
            a function that takes a vector and returns the Hamiltonian vector product
    """
    nsite = t.shape[0]
    nmode = g.shape[0]

    assert t.shape == (nsite, nsite)
    assert g.shape == (nmode, nsite, nsite)
    assert w.shape == (nmode,)

    shape = epcc.fci.make_shape(nsite, nelec, nmode, nph_max, e_only=False)

    def hop(v):
        c = v.reshape(shape)
        hc = contract_1e(t, c, nsite, nelec, nmode, nph_max, e_only=False, space="r")
        hc += contract_ep(g, c, nsite, nelec, nmode, nph_max)
        hc += contract_pp(w, c, nsite, nelec, nmode, nph_max, xi=None)
        return hc.reshape(-1)

    return hop


def build_hm(t: numpy.ndarray, g: numpy.ndarray, w: numpy.ndarray,
             nelec: ElectronSpinNumber = (1, 0), nph_max: int = 4):
    r"""
    This function generates the Hamiltonian matrix in the Fock space:
        H c_I |\Psi_I\rangle = s_{IJ} c_J |\Psi_J\rangle

    Args:
        t: numpy.ndarray (nsite, nsite)
            hopping matrix
        g: numpy.ndarray (nmode, nsite, nsite)
            electron-phonon coupling matrix
        w: numpy.ndarray (nmode, )
            phonon frequency matrix
        nelec: ElectronSpinNumber
            number of electrons
        nph_max: int
            maximum number of phonons

    Returns:
        hm: numpy.ndarray (v_shape, v_shape)
            Hamiltonian matrix
    """
    nsite = t.shape[0]
    nmode = g.shape[0]

    assert t.shape == (nsite, nsite)
    assert g.shape == (nmode, nsite, nsite)
    assert w.shape == (nmode,)

    shape = epcc.fci.make_shape(nsite, nelec, nmode, nph_max, e_only=False)
    size = prod(shape)

    # h2e_ip = fci_obj.absorb_h1e(h1e, eri, norb, nelec_ip, fac=0.5)
    if size * size * 8 / 1024 ** 3 > MAX_MEMORY_GB:
        log = pyscf.lib.logger.Logger(sys.stdout, 4)
        log.warn("Required memory in GB %6.4f", size * size * 8 / 1024 ** 3)
        raise ValueError("Not enough memory for EPH-FCI Hamiltonian.")

    hop = gen_hop(t, g, w, nelec=nelec, nph_max=nph_max)
    hm = numpy.array([hop(v) for v in numpy.eye(size)])
    return hm


def solve_fci_slow(t: numpy.ndarray, g: numpy.ndarray, w: numpy.ndarray,
                   nelec: ElectronSpinNumber = (1, 0), nph_max: int = 4,
                   nroots: int = 1):
    """
    Solve the FCI problem using the slow method.

    Args:
        t: numpy.ndarray (nsite, nsite)
            hopping matrix
        g: numpy.ndarray (nmode, nsite, nsite)
            electron-phonon coupling matrix
        w: numpy.ndarray (nmode, )
            phonon frequency matrix
        nelec: ElectronSpinNumber
            number of electrons
        nph_max: int
            maximum number of phonons

    Returns:
        e: numpy.ndarray (nroots, )
            eigenvalues
        v: numpy.ndarray (v_shape, nroots)
            eigenvectors
    """
    hm = build_hm(t, g, w, nelec=nelec, nph_max=nph_max)
    e, v = scipy.linalg.eigh(hm)
    return e[:nroots], v[:, :nroots]

def solve_fci(t: numpy.ndarray, g: numpy.ndarray, w: numpy.ndarray,
              nelec: ElectronSpinNumber = (1, 0), nph_max: int = 4,
              nroots: int = 1):
    """
    Solve the FCI problem using the fast method.

    Args:
        t: numpy.ndarray (nsite, nsite)
            hopping matrix
        g: numpy.ndarray (nmode, nsite, nsite)
            electron-phonon coupling matrix
        w: numpy.ndarray (nmode, )
            phonon frequency matrix
        nelec: ElectronSpinNumber
            number of electrons
        nph_max: int
            maximum number of phonons

    Returns:
        e: numpy.ndarray (nroots, )
            eigenvalues
        v: numpy.ndarray (v_shape, nroots)
            eigenvectors
    """
    shape = epcc.fci.make_shape(t.shape[0], nelec, g.shape[0], nph_max, e_only=False)
    size = prod(shape)
    hop = scipy.sparse.linalg.LinearOperator((size, size), matvec=gen_hop(t, g, w, nelec=nelec, nph_max=nph_max))

    e, v = scipy.sparse.linalg.eigsh(hop, k=nroots, which="SA")
    e, v = (e[0], v[:, 0]) if nroots == 1 else (e, v)
    return e, v


def dump_hol_model(hol_obj):
    """
    Dump the information from the HolModel object
    to the format that can be used by the FCI code.

    Args:
        hol_obj: HolModel
            The HolModel object.

    Returns:
        t: numpy.ndarray (nsite, nsite)
            hopping matrix
        g: numpy.ndarray (nmode, nsite, nsite)
            electron-phonon coupling matrix
        w: numpy.ndarray (nmode, )
            phonon frequency matrix
        nelec: ElectronSpinNumber
            number of electrons
    """

    # The hopping matrix
    t = hol_obj.tmatS()
    # The electron-phonon coupling matrix
    g = hol_obj.gmat[:, :hol_obj.L, :hol_obj.L].transpose(1, 2, 0)
    # The phonon frequency matrix
    w = numpy.ones([hol_obj.M]) * hol_obj.w

    # The number of electrons
    nelec = (hol_obj.na, hol_obj.nb)

    return t, g, w, nelec

def eph_fcigf_ip(hol_obj, omegas=None, ps=None, qs=None, nph_max=4, eta=0.01,
                 conv_tol=1e-8, max_cycle=100, m=40, method=None,
                 verbose=0, stdout=sys.stdout):
    """
    Compute the electron-phonon Green's function using the FCI method.
    NOTE: the number of electrons will be set to the order in which the
    number of alpha electron is greater than or equal to the number of
    beta electrons; the IP process will remove the alpha electron.

    Parameters:
        hol_obj : HolModel
            The HolModel object.
        omegas : float
            Frequency for which the Green's function is computed.
        nph_max : int, optional
            Maximum number of phonons in the system.
        ps : list of ints, optional
            List of indices of the initial states.
        qs : list of ints, optional
            List of indices of the final states.
        eta : float, optional
            Broadening factor for numerical stability.
        conv_tol : float, optional
            Convergence tolerance for the FCI solver.

    Returns:
        gfns_ip : 3D array of complex floats, shape (nomega, np, nq)
            The computed Green's function values.
    """
    log = pyscf.lib.logger.Logger(stdout, verbose)
    cput0 = (pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter())

    if method == "slow":
        log.info("\nSolving the IP Green's function by building the full FCI Hamiltonian.")
    else:
        log.info("\nSolving the IP Green's function.")

    t: numpy.ndarray
    g: numpy.ndarray
    w: numpy.ndarray
    nelec: ElectronSpinNumber
    t, g, w, nelec = dump_hol_model(hol_obj)

    # For the IP problem, we need to remove one electron from the system
    # make sure the alpha electron is greater or equal to the beta electron.
    assert nelec[0] >= nelec[1]

    nelec_ip = (nelec[0] - 1, nelec[1])
    assert nelec_ip[0] >= 0 and nelec_ip[1] >= 0

    nsite: int
    nmode: int
    nsite = t.shape[0]
    nmode = g.shape[0]

    assert t.shape == (nsite, nsite)
    assert g.shape == (nmode, nsite, nsite)
    assert w.shape == (nmode,)

    # Extract the information about the frequency and orbitals
    if ps is None:
        ps = numpy.arange(nsite)
    ps = numpy.asarray(ps)
    np = len(ps)

    if qs is None:
        qs = numpy.arange(nsite)
    qs = numpy.asarray(qs)
    nq = len(qs)

    omegas = numpy.asarray(omegas)
    nomega = len(omegas)

    # Compute the FCI solution
    ene_fci, v_fci = solve_fci(t, g, w, nelec=nelec, nph_max=nph_max, nroots=1)

    # Set up the FCI Hamiltonian
    shape = epcc.fci.make_shape(nsite, nelec, nmode, nph_max, e_only=False)
    size = prod(shape)
    hdiag = epcc.fci.make_hdiag(t, 0.0, g, w, nsite, nelec, nmode, nph_max, e_only=False, space='r')
    assert v_fci.shape == (size,)
    assert hdiag.shape == (size,)

    ndet_alph, ndet_beta = shape[:2]
    c_fci = v_fci.reshape(ndet_alph * ndet_beta, -1)

    # Set up the IP manybody Hamiltonian
    shape_ip = epcc.fci.make_shape(nsite, nelec_ip, nmode, nph_max, e_only=False)
    size_ip = prod(shape_ip)
    hdiag_ip = epcc.fci.make_hdiag(t, 0.0, g, w, nsite, nelec_ip, nmode, nph_max, e_only=False, space='r')
    assert hdiag_ip.shape == (size_ip,)

    # Build the RHS and LHS of the response equation
    bps = [[pyscf.fci.addons.des_a(c, nsite, nelec, p) for p in ps] for c in c_fci.T]
    eqs = [[pyscf.fci.addons.des_a(c, nsite, nelec, q) for q in qs] for c in c_fci.T]
    bps = numpy.asarray(bps).transpose((1, 2, 3, 0)).reshape(np, size_ip)
    eqs = numpy.asarray(eqs).transpose((1, 2, 3, 0)).reshape(nq, size_ip)

    cpu1 = log.timer("setup the FCI-GF problem", *cput0)

    if method == "slow":
        log.info("The size of the FCI Hamiltonian is %4.2f GB.", size_ip * size_ip * 8 / 1024 ** 3)

        h_ip = build_hm(t, g, w, nelec_ip, nph_max)
        assert h_ip.shape == (size_ip, size_ip)

        def gen_gfn(omega):
            omega_e0_eta_ip = omega - ene_fci - 1j * eta
            h_ip_omega = h_ip + omega_e0_eta_ip * numpy.eye(size_ip)
            xps = numpy.linalg.solve(h_ip_omega, bps.T).T
            return numpy.dot(xps, eqs.T)

    else:
        hop_ip = gen_hop(t, g, w, nelec_ip, nph_max)

        def gen_gfn(omega):
            omega_e0_eta_ip = omega - ene_fci - 1j * eta
            hdiag_ip_omega = hdiag_ip + omega_e0_eta_ip

            def h_ip_omega(v):
                assert v.shape == (size_ip,)
                hv_real = hop_ip(v.real)
                hv_imag = hop_ip(v.imag)

                hv = hv_real + 1j * hv_imag
                hv += omega_e0_eta_ip * v

                return hv.reshape((size_ip,))

            xps = gmres(h_ip_omega, bs=bps, xs0=bps / hdiag_ip_omega,
                        diag=hdiag_ip_omega, tol=conv_tol,
                        max_cycle=max_cycle, m=m,
                        verbose=verbose, stdout=stdout)

            xps = xps.reshape(np, size_ip)
            return numpy.dot(xps, eqs.T)

    cpu1 = log.timer("initialize the Hamiltonian", *cpu1)

    # Solve the Green's function
    gfns_ip = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape((nomega, np, nq))
    cpu1 = log.timer("solve the FCI-GF problem", *cpu1)

    # Return the Green's function
    return gfns_ip

def eph_fcigf_ea(hol_obj, omegas=None, ps=None, qs=None, nph_max=4, eta=0.01,
                 conv_tol=1e-8, max_cycle=100, m=40, method=None,
                 verbose=0, stdout=sys.stdout):
    """
    Compute the electron-phonon Green's function using the FCI method.

    Parameters:
        hol_obj : HolModel
            The HolModel object.
        omegas : float
            Frequency for which the Green's function is computed.
        nph_max : int, optional
            Maximum number of phonons in the system.
        ps : list of ints, optional
            List of indices of the initial states.
        qs : list of ints, optional
            List of indices of the final states.
        eta : float, optional
            Broadening factor for numerical stability.
        conv_tol : float, optional
            Convergence tolerance for the FCI solver.

    Returns:
        gfns_ea : 3D array of complex floats, shape (nomega, np, nq)
            The computed Green's function values.
    """
    log = pyscf.lib.logger.Logger(stdout, verbose)
    cput0 = (pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter())

    if method == "slow":
        log.info("\nSolving the EA Green's function by building the full FCI Hamiltonian.")
    else:
        log.info("\nSolving the EA Green's function.")

    t: numpy.ndarray
    g: numpy.ndarray
    w: numpy.ndarray
    nelec: ElectronSpinNumber
    t, g, w, nelec = dump_hol_model(hol_obj)

    # For the EA problem, we need to add one electron to the system
    # make sure the alpha electron is greater than or equal to the beta electron.
    # TODO: check this, it might be wrong
    assert nelec[0] >= nelec[1]

    nelec_ea = (nelec[0], nelec[1]+1)
    assert nelec_ea[0] >= 0 and nelec_ea[1] >= 0

    nsite: int
    nmode: int
    nsite = t.shape[0]
    nmode = g.shape[0]

    assert t.shape == (nsite, nsite)
    assert g.shape == (nmode, nsite, nsite)
    assert w.shape == (nmode,)

    # Extract the information about the frequency and orbitals
    if ps is None:
        ps = numpy.arange(nsite)
    ps = numpy.asarray(ps)
    np = len(ps)

    if qs is None:
        qs = numpy.arange(nsite)
    qs = numpy.asarray(qs)
    nq = len(qs)

    omegas = numpy.asarray(omegas)
    nomega = len(omegas)

    # Solve the FCI problem
    ene_fci, v_fci = solve_fci(t, g, w, nelec=nelec, nph_max=nph_max, nroots=1)

    # Set up the FCI Hamiltonian
    shape = epcc.fci.make_shape(nsite, nelec, nmode, nph_max, e_only=False)
    size = prod(shape)
    hdiag = epcc.fci.make_hdiag(t, 0.0, g, w, nsite, nelec, nmode, nph_max, e_only=False, space='r')
    assert v_fci.shape == (size,)
    assert hdiag.shape == (size,)

    ndet_alph, ndet_beta = shape[:2]
    c_fci = v_fci.reshape(ndet_alph * ndet_beta, -1)

    # Set up the IP manybody Hamiltonian
    shape_ea = epcc.fci.make_shape(nsite, nelec_ea, nmode, nph_max, e_only=False)
    size_ea = prod(shape_ea)
    hdiag_ea = epcc.fci.make_hdiag(t, 0.0, g, w, nsite, nelec_ea, nmode, nph_max, e_only=False, space='r')
    assert hdiag_ea.shape == (size_ea,)

    # Build the RHS and LHS of the response equation
    bps = [[pyscf.fci.addons.cre_b(c, nsite, nelec, p) for p in ps] for c in c_fci.reshape(ndet_alph * ndet_beta, -1).T]
    eqs = [[pyscf.fci.addons.cre_b(c, nsite, nelec, q) for q in qs] for c in c_fci.reshape(ndet_alph * ndet_beta, -1).T]
    bps = numpy.asarray(bps).transpose((1, 2, 3, 0)).reshape(np, size_ea)
    eqs = numpy.asarray(eqs).transpose((1, 2, 3, 0)).reshape(nq, size_ea)

    cpu1 = log.timer("setup the FCI-GF problem", *cput0)

    if method == "slow":
        log.info("The size of the FCI Hamiltonian is %4.2f GB.", size_ea * size_ea * 8 / 1024 ** 3)

        h_ea = build_hm(t, g, w, nelec_ea, nph_max)
        assert h_ea.shape == (size_ea, size_ea)

        def gen_gfn(omega):
            omega_e0_eta_ea = omega + ene_fci + 1j * eta
            h_ea_omega = - h_ea + omega_e0_eta_ea * numpy.eye(size_ea)
            xps = numpy.linalg.solve(h_ea_omega, bps.T).T
            return numpy.dot(xps, eqs.T)

    else:
        hop_ea = gen_hop(t, g, w, nelec_ea, nph_max)

        def gen_gfn(omega):
            omega_e0_eta_ea = omega + ene_fci + 1j * eta
            hdiag_ea_omega = - hdiag_ea + omega_e0_eta_ea

            def h_ea_omega(v):
                assert v.shape == (size_ea,)
                hv_real = hop_ea(v.real)
                hv_imag = hop_ea(v.imag)

                hv = - (hv_real + 1j * hv_imag)
                hv += omega_e0_eta_ea * v

                return hv.reshape((size_ea,))

            xps = gmres(h_ea_omega, bs=bps, xs0=bps / hdiag_ea_omega,
                        diag=hdiag_ea_omega, tol=conv_tol,
                        max_cycle=max_cycle, m=m,
                        verbose=verbose, stdout=stdout)

            xps = xps.reshape(np, size_ea)
            return numpy.dot(xps, eqs.T)

    cpu1 = log.timer("initialize the Hamiltonian", *cpu1)

    # Solve the Green's function
    gfns_ea = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape((nomega, np, nq))
    cpu1 = log.timer("solve the FCI-GF problem", *cpu1)

    # Return the Green's function
    return gfns_ea
