import sys, os
import numpy, scipy

import epcc.fci
from epcc.hol_model import HolModel
from epcc.fci import contract_1e
from epcc.fci import contract_ep_rspace as contract_ep
from epcc.fci import contract_pp

from ephfcigf import eph_fcigf_ip, eph_fcigf_ea

def solve(omegas, nph_max=10, m=None, log=sys.stdout, tmp=None):
    eta = 0.01
    gf1_ip = eph_fcigf_ip(m, omegas, ps=None, qs=None, eta=eta, nph_max=nph_max, verbose=5, stdout=log)
    # gf1_ea = eph_fcigf_ea(m, omegas, ps=None, qs=None, eta=eta, nph_max=nph_max, verbose=5, stdout=log)
    gf_fci = gf1_ip # + gf1_ea

    for iomega, omega in enumerate(omegas):
        s = - numpy.trace(gf_fci[:, :, iomega].imag) / numpy.pi
        log.write("omega = %f, s = %f\n" % (omega, s))

    return gf_fci

if __name__ == '__main__':
    import h5py
    from pyscf.lib import chkfile
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nomega_total = 32
    nomega_per_node = nomega_total // size
    assert nomega_per_node * size == nomega_total

    log = "%s/log-%02d.out" % (os.environ['LOG_TMPDIR'], rank)
    tmp = "%s/tmp-%02d.h5"  % (os.environ['LOG_TMPDIR'], rank)

    nsite = 4
    nmode = 4
    nelec = (1, 0)

    nph_max  = 2
    conv_tol = 1e-6

    m = HolModel(
        nsite, nmode, nelec[0] + nelec[1],
        1.0, 0.1, bc='p', gij=None,
        ca=numpy.eye(nsite), cb=None
    )
    m.na = 1
    m.nb = 0

    omegas = numpy.linspace(-0.5, 0.5, nomega_total)
    tmp    = solve(
        omegas[rank * nomega_per_node : (rank + 1) * nomega_per_node],
        nph_max=nph_max, m=m, log=open(log, 'w'), tmp=h5py.File(tmp, 'w')
        )

    print(tmp.shape)
    print("Rank %d finished." % rank)

    gf_fci = comm.gather(tmp, root=0)

    print(gf_fci)

    if rank == 0:
        gf = numpy.concatenate(gf_fci, axis=0)
        print(gf.shape)
