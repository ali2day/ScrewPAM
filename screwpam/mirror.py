"""Opposite enantiomer of a phonopy object by an exact symmetry operation (structure AND force constants), so that both
hands have identical frequencies by construction and differ only in handedness."""
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def _cart_op(lat, P):
    """Cartesian matrix of the operation r_frac -> P r_frac (row-vector lattice)."""
    return lat.T @ P @ np.linalg.inv(lat.T)


def enantiomer(ph, P=((0, 1, 0), (1, 0, 0), (0, 0, 1))):
    """Return a new Phonopy object for the image of `ph` under r_frac -> P r_frac (default: swap a1 <-> a2, a vertical
    mirror for a hexagonal cell). The operation must be orthogonal in Cartesian space. Force constants are transformed
    as Phi'(i', j') = M Phi(i, j) M^T with the supercell atom map i -> i'."""
    P = np.array(P, float)
    uc = ph.unitcell
    M = _cart_op(uc.cell, P)
    if not np.allclose(M @ M.T, np.eye(3), atol=1e-8):
        raise ValueError("operation is not orthogonal for this lattice")
    new_uc = PhonopyAtoms(symbols=uc.symbols, cell=uc.cell, scaled_positions=(uc.scaled_positions @ P.T) % 1.0,
                          masses=uc.masses)
    out = Phonopy(new_uc, supercell_matrix=ph.supercell_matrix, primitive_matrix=ph.primitive_matrix,
                  symprec=ph.symmetry.tolerance)
    sc0, sc1 = ph.supercell, out.supercell
    # supercell atoms: image of atom i of the old supercell, expressed in the (identical) supercell lattice
    img = (sc0.scaled_positions @ (np.linalg.inv(sc0.cell.T) @ M @ sc0.cell.T).T) % 1.0
    perm = np.zeros(len(img), int)
    for i, p in enumerate(img):
        d = sc1.scaled_positions - p
        d -= np.rint(d)
        j = np.argmin(np.linalg.norm(d @ sc1.cell, axis=1))
        if np.linalg.norm(d[j] @ sc1.cell) > 1e-4:
            raise ValueError("supercell atom %d has no image" % i)
        perm[i] = j
    fc = ph.force_constants
    if fc is None:
        raise ValueError("force constants of the input object are not set")
    fc_new = np.zeros_like(fc)
    fc_new[np.ix_(perm, perm)] = np.einsum("ab,ijbc,dc->ijad", M, fc, M)
    out.force_constants = fc_new
    return out
