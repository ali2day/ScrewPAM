"""Tellurium benchmark: reproduces the pseudo-angular momentum assignment of Ishito et al., Chirality 35, 338 (2023)
(arXiv:2212.02720): "In R-Te (P3_121) the phonon of the split Gamma_3 mode with a higher frequency has m = -1, whereas
in L-Te (P3_221) it has m = +1."

Input (this folder): POSCAR (R-Te, P3_121, lattice of Ishito et al., not relaxed) and FORCE_SETS
(VASP 6.4.2, PBE, no SOC, PAW Te, 400 eV, 3x3x3 supercell, 4x4x4 k-mesh, +-0.01 A).
L-Te is generated from R-Te by the exact vertical mirror a1 <-> a2 (structure and force constants).

    python run_te.py
"""
import os

import numpy as np
import phonopy

import screwpam as sp

HERE = os.path.dirname(os.path.abspath(__file__))
CM = 33.35641
QZ = 0.01            # small q along Gamma -> A (the Raman wave vector of Ishito et al. is ~1.4 % of Gamma-A)


def load_r():
    ph = phonopy.load(supercell_matrix=[3, 3, 3], unitcell_filename=os.path.join(HERE, "POSCAR"),
                      force_sets_filename=os.path.join(HERE, "FORCE_SETS"), is_compact_fc=False,
                      symmetrize_fc=True, log_level=0)
    return ph


def table(ph, label):
    qz = np.array([0.0, QZ])
    f, E = sp.modes_along_c(ph, qz)
    lz, _ = sp.angular_momentum(E)
    m, mag, op = sp.pseudo_angular_momentum(ph, qz, E)
    sg = ph.symmetry.get_international_table()
    print("\n%s  (%s, screw {C%d+|%.4f c})" % (label, sg, op["n"], op["tau"][2]))
    print("  Gamma frequencies (cm-1): " + ", ".join("%.1f" % x for x in f[0] * CM))
    print("  q_z = %.2f:   freq (cm-1)   l_z (hbar)    m   |lambda|" % QZ)
    for k in range(f.shape[1]):
        print("               %8.2f      %+6.2f     %+d    %.3f" % (f[1, k] * CM, lz[1, k], m[1, k], mag[1, k]))
    return f * CM, lz, m


def gamma3_upper_m(f, m):
    """m of the higher-frequency branch of each Gamma_3 (E) doublet at q_z = QZ."""
    g = f[0]
    out = []
    for k in range(len(g) - 1):
        if g[k] > 1 and abs(g[k + 1] - g[k]) < 0.05:                 # degenerate doublet at Gamma
            hi = k if f[1, k] > f[1, k + 1] else k + 1
            out.append((g[k], m[1, hi]))
    return out


if __name__ == "__main__":
    phR = load_r()
    phL = sp.enantiomer(phR)
    fR, lR, mR = table(phR, "R-Te")
    fL, lL, mL = table(phL, "L-Te (mirror image)")
    print("\nIshito et al. (2023): higher branch of the split Gamma_3 doublet has m = -1 (R-Te) and m = +1 (L-Te)")
    ok = True
    for (gR, uR), (gL, uL) in zip(gamma3_upper_m(fR, mR), gamma3_upper_m(fL, mL)):
        res = (uR == -1) and (uL == +1)
        ok &= res
        print("  Gamma_3 at %6.1f cm-1:  R-Te m = %+d,  L-Te m = %+d   %s" % (gR, uR, uL, "OK" if res else "DIFFERENT"))
    print("\nexperiment (Raman, Ishito et al. Table 1): Gamma_3 93, Gamma_1 121, Gamma_3 141 cm-1")
    print("max |f_R - f_L| = %.2e cm-1" % np.abs(fR - fL).max())
    print("RESULT:", "reproduced" if ok else "NOT reproduced")
