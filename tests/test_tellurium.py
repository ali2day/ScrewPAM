"""pytest: checks of ScrewPAM on the tellurium example (run from the repository root: pytest)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "tellurium"))
import screwpam as sp                      # noqa: E402
import run_te                              # noqa: E402


@pytest.fixture(scope="module")
def hands():
    phR = run_te.load_r()
    return phR, sp.enantiomer(phR)


def _modes(ph, qz):
    f, E = sp.modes_along_c(ph, qz)
    lz, _ = sp.angular_momentum(E)
    m, mag, op = sp.pseudo_angular_momentum(ph, qz, E)
    return f, lz, m, mag, op


def test_screw_found(hands):
    for ph, tau in zip(hands, (1 / 3, 2 / 3)):
        op = sp.find_screw(ph)
        assert op["n"] == 3 and abs(op["tau"][2] % 1 - tau) < 1e-6


def test_clean_eigenstates(hands):
    qz = np.linspace(-0.49, 0.49, 50)
    for ph in hands:
        *_, mag, _ = _modes(ph, qz)
        assert (mag > 0.99).all()


def test_acoustic(hands):
    """rigid translations: z -> m = 0, circular in-plane -> m = sign(l_z)."""
    for ph in hands:
        f, E = sp.modes_along_c(ph, np.array([0.02]))
        lz, _ = sp.angular_momentum(E)
        m, _, _ = sp.pseudo_angular_momentum(ph, np.array([0.02]), E)
        for k in range(3):
            zfrac = (np.abs(E[0, k, :, 2]) ** 2).sum()
            expect = 0 if zfrac > 0.9 else int(np.sign(lz[0, k]))
            assert m[0, k] == expect


def test_time_reversal(hands):
    qz = np.array([0.1, -0.1])
    for ph in hands:
        f, lz, m, mag, _ = _modes(ph, qz)
        assert np.allclose(f[0], f[1], atol=1e-6)
        gap = np.minimum(np.abs(np.diff(f[0], prepend=-1)), np.abs(np.diff(f[0], append=99)))
        ok = gap > 1e-4                                   # non-degenerate modes: same ordering at +q and -q
        assert ((m[0][ok] + m[1][ok]) % 3 == 0).all()
        assert np.allclose(lz[0][ok], -lz[1][ok], atol=1e-6)


def test_mirror(hands):
    qz = np.linspace(0.02, 0.48, 12)
    (fR, lR, mR, _, _), (fL, lL, mL, _, _) = (_modes(ph, qz) for ph in hands)
    assert np.allclose(fR, fL, atol=1e-6)
    assert np.allclose(lR, -lL, atol=1e-6)
    assert ((mR + mL) % 3 == 0).all()


def test_ishito_2023(hands):
    """Ishito et al., Chirality 35, 338 (2023): higher branch of the split Gamma_3 doublet has m = -1 (R), +1 (L)."""
    res = []
    for ph in hands:
        f, lz, m, _, _ = _modes(ph, np.array([0.0, run_te.QZ]))
        res.append(run_te.gamma3_upper_m(f * run_te.CM, m))
    assert len(res[0]) == 2
    assert all(u == -1 for _, u in res[0])
    assert all(u == +1 for _, u in res[1])
