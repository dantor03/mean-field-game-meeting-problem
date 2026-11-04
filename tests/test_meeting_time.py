import numpy as np
import pytest

import meeting_time_mfg as mfg
from meeting_time_mfg import Params, solve_fp, solve_hjb, solve_meeting_time


@pytest.fixture(scope="module")
def results():
    return {s: solve_meeting_time(scheme=s) for s in mfg.SCHEMES}


def test_arrival_cost():
    p = Params()
    assert mfg.arrival_cost(0.3, 1.0, p) == pytest.approx(p.gamma * 0.7)        # waits
    assert mfg.arrival_cost(0.8, 1.0, p) == pytest.approx(p.alpha * 0.3 + p.gamma * 0.2)
    assert mfg.arrival_cost(1.2, 1.0, p) == pytest.approx(p.alpha * 0.7 + p.beta * 0.2)


@pytest.mark.parametrize("scheme", mfg.SCHEMES)
def test_fixed_point(scheme, results):
    res, p = results[scheme], Params(nt=mfg.DEFAULT_NT[scheme])
    assert res.converged
    assert abs(mfg.quorum_time(res.arrivals, p) - res.T) < 5e-3
    assert p.t_scheduled < res.T < p.t_final


def test_explicit_and_implicit_schemes_agree(results):
    assert results["implicit"].T == pytest.approx(results["explicit"].T, abs=0.03)


def test_implicit_scheme_converges_to_the_explicit_one(results):
    fine = solve_meeting_time(Params(nt=1601), scheme="implicit")
    coarse = results["implicit"]
    T = results["explicit"].T
    assert abs(fine.T - T) < abs(coarse.T - T)
    assert fine.T == pytest.approx(T, abs=5e-3)


@pytest.mark.parametrize("scheme", mfg.SCHEMES)
def test_density_stays_a_probability_and_arrivals_are_monotone(scheme, results):
    res = results[scheme]
    assert res.m.min() >= 0.0
    assert res.arrivals[0] == 0.0
    assert np.all(np.diff(res.arrivals) >= -1e-12)
    assert res.arrivals[-1] <= 1.0 + 1e-12


@pytest.mark.parametrize("scheme", mfg.SCHEMES)
def test_no_mass_crosses_the_wall(scheme):
    # Everybody walks away from the room and there is no diffusion: mass piles
    # up at the wall, and none may appear, disappear or reach the room.
    p = Params(nt=mfg.DEFAULT_NT[scheme], sigma=0.0)
    m, arrivals = solve_fp(np.full((p.nt, p.nx), -1.0), p, scheme)
    assert np.abs(arrivals).max() < 1e-12
    assert m.min() >= 0.0
    assert m[-1, 0] > m[0, 0]


@pytest.mark.parametrize("scheme", mfg.SCHEMES)
def test_hjb_boundary_conditions(scheme):
    p = Params(nt=mfg.DEFAULT_NT[scheme])
    T = 1.3
    u = solve_hjb(T, p, scheme)
    assert np.allclose(u[:, -1], mfg.arrival_cost(p.t, T, p))
    assert np.allclose(u[:-1, 0] - u[:-1, 1], p.wall_slope * p.dx)
    assert np.isfinite(u).all()


def test_unknown_scheme():
    with pytest.raises(ValueError):
        solve_hjb(1.0, Params(nt=11), "crank-nicolson")


def test_command_line_writes_the_figures(tmp_path):
    res = mfg.main(["--scheme", "implicit", "--no-animation", "--out", str(tmp_path)])
    assert res.converged
    assert (tmp_path / "meeting_time.png").exists()
    assert (tmp_path / "meeting_time_3d.png").exists()
