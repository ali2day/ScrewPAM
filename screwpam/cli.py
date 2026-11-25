"""Command line: screwpam phonopy_params.yaml [--nq 201] [--out pam.dat]
Writes q_z, frequency (cm^-1), l_z (hbar), pseudo-angular momentum m and |lambda| for every mode along -A..Gamma..A."""
import argparse

import numpy as np
import phonopy

from .core import angular_momentum, modes_along_c, pseudo_angular_momentum, qpoints_gamma_a

THZ_TO_CM = 33.35641


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phonopy_file", help="phonopy yaml with force constants or forces (e.g. phonopy_params.yaml)")
    p.add_argument("--nq", type=int, default=201, help="number of q points from -A to A (default 201)")
    p.add_argument("--out", default="pam.dat", help="output table (default pam.dat)")
    a = p.parse_args(argv)
    ph = phonopy.load(a.phonopy_file, log_level=0)
    qz = qpoints_gamma_a(a.nq)
    f, E = modes_along_c(ph, qz)
    lz, _ = angular_momentum(E)
    m, mag, op = pseudo_angular_momentum(ph, qz, E)
    nm = f.shape[1]
    np.savetxt(a.out, np.column_stack([np.repeat(qz, nm), f.ravel() * THZ_TO_CM, lz.ravel(), m.ravel(), mag.ravel()]),
               fmt=["%10.5f", "%12.4f", "%9.4f", "%4d", "%8.4f"],
               header="screw {C%d+|tau_z=%.4f}; m defined by lambda' = exp(-i 2 pi m / %d)\n"
                      "q_z(2pi/c)  freq(cm-1)  l_z(hbar)  m  |lambda|" % (op["n"], op["tau"][2], op["n"]))
    inner = (np.abs(qz) > 1e-9) & (np.abs(np.abs(qz) - 0.5) > 1e-9)
    print("screw {C%d+|tau_z=%.4f}; %d q points; clean eigenstates (|lambda|>0.99) on the open line: %.1f %%; written %s"
          % (op["n"], op["tau"][2], a.nq, 100 * (mag[inner] > 0.99).mean(), a.out))


if __name__ == "__main__":
    main()
