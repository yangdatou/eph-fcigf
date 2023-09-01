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

    eta = 0.8
    # gf1_ip = eph_fcigf_ip(m, omegas, ps=None, qs=None, eta=eta, max_cycle=10, conv_tol=1e-2, nph_max=nph_max, verbose=5, stdout=log)
    gf1_ea = eph_fcigf_ea(m, omegas, ps=None, qs=None, eta=eta, max_cycle=40, conv_tol=1e-2, nph_max=nph_max, verbose=5, stdout=log)
    gf_fci = gf1_ea

    log.write("\n")
    for iomega, omega in enumerate(omegas):
        s = - numpy.trace(gf_fci[iomega, :, :].imag) / numpy.pi
        log.write("omega = % 12.8f, s = % 12.8f\n" % (omega, s))

    log.write("\n")
    log.write("Wall time to solve the Green's function: % 6.4f min\n" % ((time.time() - t0) / 60.0))

    return gf_fci

if __name__ == '__main__':
    # from mpi4py import MPI
    #
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # nomega_total = 26
    for nph_max in [2, 4, 6]:
        for g in [1.1, 1.5, 2.0]:
            nomega = 26 # nomega_total // size
            # assert nomega * size == nomega_total

            log = "/Users/yangjunjie/work/cc-eph/eph-fcigf/out/tmp/log.out"

            nsite = 6
            nmode = 6
            nelec = (1, 0)

            m = HolModel(
                nsite, nmode, nelec[0] + nelec[1],
                1.0, g, bc='p', gij=None,
                ca=numpy.eye(nsite),
                cb=numpy.eye(nsite)
            )
            m.na = 1
            m.nb = 0

            omegas = numpy.linspace(-10.0, 0.0, nomega)
            gf_fci = solve(-omegas, nph_max=nph_max, m=m, log=sys.stdout)
            assert gf_fci.shape == (nomega, nsite, nsite)

            label = "CC-GF" if nph_max == 2 else "FCI-GF(%d)" % (2 ** (nph_max-2))
            f = f"./out/gf-{label}-g-{g}-2.out"
            with open(f, 'w') as f:
                for iomega, omega in enumerate(omegas):
                    s = numpy.trace(gf_fci[iomega, :, :].imag) / numpy.pi
                    print("% 6.4f, % 12.8f" % (omega, s))
                    f.write("% 6.4f, % 12.8f\n" % (omega, s))
