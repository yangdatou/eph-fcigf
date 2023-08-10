import sys, os
import numpy, scipy

import epcc.fci
from epcc.hol_model import HolModel
from epcc.fci import contract_1e
from epcc.fci import contract_ep_rspace as contract_ep
from epcc.fci import contract_pp


from ephfcigf import eph_fcigf_ip, eph_fcigf_ea

def main():
    nsite = 4
    nmode = 4
    nelec = (1, 0)

    nph_max  = 20
    conv_tol = 1e-6

    m = HolModel(
        nsite, nmode, nelec[0] + nelec[1],
        1.0, 0.1, bc='p', gij=None,
        ca=numpy.eye(nsite), cb=None
    )
    m.na = 1
    m.nb = 0

    eta = 0.01
    ps = None # [0, 1, 2, 3]
    qs = None # [0, 1, 2, 3]
    omegas = numpy.linspace(-0.5, 0.5, 21)

    gf1_ip = eph_fcigf_ip(m, omegas, ps=ps, qs=qs, eta=eta, nph_max=nph_max, verbose=5, stdout=sys.stdout)

    # assert 1 == 2
    # gf1_ea = eph_fcigf_ea(m, omegas, ps=ps, qs=qs, eta=eta, nph_max=nph_max, verbose=5, stdout=sys.stdout)
    # gf_fci = gf1_ip + gf1_ea
    #
    # # gf2_ip = eph_fcigf_ip(m, omegas, ps=ps, qs=qs, eta=eta, nph_max=nph_max, method="slow", conv_tol=conv_tol, verbose=5, stdout=sys.stdout)
    # # err_ip = numpy.linalg.norm(gf1_ip - gf2_ip)
    # # assert err_ip < conv_tol
    # #
    # # gf2_ea = eph_fcigf_ea(m, omegas, ps=ps, qs=qs, eta=eta, nph_max=nph_max, method="slow", conv_tol=conv_tol, verbose=5, stdout=sys.stdout)
    # # err_ea = numpy.linalg.norm(gf1_ea - gf2_ea)
    # # assert err_ea < conv_tol
    #
    # for iomega, omega in enumerate(omegas):
    #     s = - numpy.trace(gf_fci[:, :, iomega].imag) / numpy.pi
    #     print("% 8.4f, %8.4f" % (omega, s))

if __name__ == '__main__':
    numpy.show_config()
    main()