import sys, os, time
import numpy, scipy

import epcc.fci
from epcc.hol_model import HolModel
from epcc.fci import contract_1e
from epcc.fci import contract_ep_rspace as contract_ep
from epcc.fci import contract_pp

from ephfcigf import eph_fcigf_ip, eph_fcigf_ea

def solve(omegas, nph_max=10, m=None, log=sys.stdout, tmp=None):
    t0 = time.time()
    log.write("nph_max = %d\n" % nph_max)
    log.write("omegas  = \n" % omegas)
    for omega in omegas:
        log.write("% 6.4f\n" % omega)
    log.write("\n")

    eta = 0.08
    # gf1_ip = eph_fcigf_ip(m, omegas, ps=None, qs=None, eta=eta, conv_tol=1e-4, nph_max=nph_max, verbose=5, stdout=log)
    gf1_ea = eph_fcigf_ea(m, omegas, ps=None, qs=None, eta=eta, conv_tol=1e-4, nph_max=nph_max, verbose=5, stdout=log)
    gf_fci = gf1_ea

    log.write("\n")
    for iomega, omega in enumerate(omegas):
        s = - numpy.trace(gf_fci[iomega, :, :].imag) / numpy.pi
        log.write("omega = % 12.8f, s = % 12.8f\n" % (omega, s))

    log.write("\n")
    log.write("Wall time to solve the Green's function: % 6.4f min\n" % ((time.time() - t0) / 60.0))

    return gf_fci

if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nomega_total = 128
    nomega = nomega_total // size
    assert nomega * size == nomega_total

    log = "%s/%02d.log" % (os.environ['LOG_TMPDIR'], rank)

    nsite = 6
    nmode = 6
    nelec = (1, 0)

    nph_max  = 16

    m = HolModel(
        nsite, nmode, nelec[0] + nelec[1],
        1.0, 1.1, bc='p', gij=None,
        ca=numpy.eye(nsite), cb=None
    )
    m.na = 1
    m.nb = 0

    omegas = numpy.linspace(-2.0, 6.0, nomega_total).reshape(size, nomega)
    res    = solve(omegas[rank], nph_max=nph_max, m=m, log=open(log, 'w'))
    assert res.shape == (nomega, nsite, nsite)

    tmp = comm.gather(res, root=0)

    if rank == 0:
        gf_fci = numpy.concatenate(tmp, axis=0)
        assert gf_fci.shape == (nomega_total, nsite, nsite)

        with open("gf.out", 'w') as f:
            for iomega, omega in enumerate(omegas.reshape(-1)):
                s = - numpy.trace(gf_fci[iomega, :, :].imag) / numpy.pi
                print("omega = % 6.4f, s = % 12.8f" % (omega, s))
                f.write("% 6.4f, % 12.8f\n" % (omega, s))
