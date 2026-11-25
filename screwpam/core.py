"""Phonon angular momentum and screw-axis pseudo-angular momentum (PAM) from a phonopy object.

Conventions
-----------
* Phonopy eigenvectors: u_k(l, t) ~ e_k exp(i q.r(kl) - i omega t), r in fractional coordinates.
* Angular momentum of a mode (Zhang & Niu, PRL 112, 085503 (2014)):
      l_z = hbar * sum_k e_k^dagger M_z e_k,   (M_z)_{ab} = -i eps_{zab}
  +1 (in units of hbar) = counter-clockwise circular motion seen from +z.
* Screw operation g = {C_n^+ | tau}: rotation by +2 pi / n about z followed by the translation tau (tau_z = c * p / n).
  Along q = (0, 0, q_z) every non-degenerate mode is an eigenstate, g u = lambda u. Removing the trivial phase of the
  translation, lambda' = lambda * exp(i 2 pi q.tau) = exp(-i 2 pi m / n) defines the pseudo-angular momentum
  m in (-n/2, n/2] (Zhang & Murakami, PRR 4, L012024 (2022)).
  With this convention a transverse acoustic mode with l_z = +1 has m = +1 and the longitudinal one m = 0.
"""
import numpy as np

MZ = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])


def qpoints_gamma_a(nq=201):
    """q_z grid from -A to A (fractional, units of the reciprocal vector b3)."""
    return np.linspace(-0.5, 0.5, nq)


def modes_along_c(ph, qz):
    """Frequencies (THz) and eigenvectors (nq, nmodes, natoms, 3) of a phonopy object along (0, 0, q_z)."""
    ph.run_qpoints([[0, 0, x] for x in qz], with_eigenvectors=True)
    res = getattr(ph, "qpoints", None)
    if res is not None and hasattr(res, "eigenvectors"):          # phonopy >= 2.4x
        f, ev = res.frequencies, res.eigenvectors
    else:                                                          # older phonopy
        d = ph.get_qpoints_dict()
        f, ev = d["frequencies"], d["eigenvectors"]
    nat = len(ph.primitive.numbers)
    E = np.array([ev[i].T.reshape(f.shape[1], nat, 3) for i in range(len(qz))])
    return f, E


def angular_momentum(E):
    """l_z (units of hbar) of every mode; E = (nq, nmodes, natoms, 3). Returns (total (nq, nmodes), per atom)."""
    per_atom = np.real(np.einsum("qmai,ij,qmaj->qma", E.conj(), MZ, E))
    return per_atom.sum(-1), per_atom


def find_screw(ph, symprec=None):
    """Screw (or pure rotation) about z with the smallest positive angle among the symmetry operations of the
    primitive cell.  Returns dict(R (fractional), Rc (Cartesian), tau (fractional), n, angle)."""
    prim = ph.primitive
    lat = prim.cell
    ops = ph.primitive_symmetry.symmetry_operations
    best = None
    for R, t in zip(ops["rotations"], ops["translations"]):
        Rc = lat.T @ R @ np.linalg.inv(lat.T)
        if abs(np.linalg.det(Rc) - 1) > 1e-6 or abs(Rc[2, 2] - 1) > 1e-6:
            continue                                   # proper rotations about z only
        ang = np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0]))
        if ang > 1e-3 and (best is None or ang < best[0]):
            best = (ang, R, Rc, t)
    if best is None:
        raise ValueError("no proper rotation about z with positive angle in the symmetry group")
    ang, R, Rc, tau = best
    return dict(R=R, Rc=Rc, tau=np.array(tau, float), n=int(round(360.0 / ang)), angle=ang)


def atom_map(ph, op):
    """Permutation k -> j and lattice vector L with R r_k + tau = r_j + L (fractional)."""
    prim = ph.primitive
    lat, r = prim.cell, prim.scaled_positions
    nat = len(r)
    perm, L = np.zeros(nat, int), np.zeros((nat, 3))
    for k in range(nat):
        p = op["R"] @ r[k] + op["tau"]
        d = p - r
        j = np.argmin(np.linalg.norm((d - np.rint(d)) @ lat, axis=1))
        if np.linalg.norm((d[j] - np.rint(d[j])) @ lat) > 1e-3:
            raise ValueError("atom %d is not mapped onto an atom by the screw operation" % k)
        perm[k], L[k] = j, np.rint(p - r[j])
    return perm, L


def pseudo_angular_momentum(ph, qz, E, op=None):
    """PAM m (int), |lambda| (1 for a clean eigenstate) of every mode on (0, 0, q_z); also returns the operation.
    Modes at degenerate points (e.g. Gamma, A) need not be eigenstates: check |lambda| > 0.99 before using m."""
    op = op or find_screw(ph)
    perm, L = atom_map(ph, op)
    r = ph.primitive.scaled_positions
    n = op["n"]
    m = np.zeros(E.shape[:2], int)
    mag = np.zeros(E.shape[:2])
    for iq, x in enumerate(qz):
        q = np.array([0.0, 0.0, x])
        ph0 = np.exp(2j * np.pi * r @ q)                            # Bloch phase of each atom in cell 0
        for im in range(E.shape[1]):
            e = E[iq, im]
            gu = np.zeros_like(e)
            gu[perm] = (op["Rc"] @ (e * ph0[:, None]).T).T * np.exp(-2j * np.pi * (L @ q))[:, None]
            lam = np.sum((e * ph0[:, None]).conj() * gu) * np.exp(2j * np.pi * q @ op["tau"])
            mag[iq, im] = abs(lam)
            mm = int(np.rint(-n * np.angle(lam) / (2 * np.pi))) % n
            m[iq, im] = mm - n if mm > n // 2 else mm
    return m, mag, op
