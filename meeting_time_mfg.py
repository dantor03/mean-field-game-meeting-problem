"""Meeting starting time mean field game (Guéant, Lasry and Lions, 2011).

A continuum of agents walks towards a meeting room at x = 0.  Each agent
controls its drift, pays a quadratic effort cost and, on arrival at time tau,
pays

    c(tau, T) = alpha (tau - t_s)^+ + beta (tau - T)^+ + gamma (T - tau)^+,

where T is the time at which a quorum of the agents has arrived.  For a guess
of T the value function u and the density m solve, with nu = sigma^2 / 2,

    HJB:  u_t + nu u_xx - (1/2) u_x^2 = 0,           u(t, 0) = c(t, T),
    FP:   m_t - nu m_xx - (m u_x)_x = 0,              m(t, 0) = 0,

with a reflecting wall at x = x_min, and T is updated by a damped fixed-point
iteration until it equals the quorum time of the density it produces.

Two time-steppers share the same spatial discretization:

    explicit   forward/backward Euler (small time step, nt = 4001)
    implicit   IMEX: implicit diffusion (sparse LU), explicit Hamiltonian and
               upwind advection (nt = 401 by default)

Run ``python meeting_time_mfg.py --help`` for the command-line options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

SCHEMES = ("explicit", "implicit")
DEFAULT_NT = {"explicit": 4001, "implicit": 401}


@dataclass(frozen=True)
class Params:
    # grid
    x_min: float = -5.0          # furthest starting position
    x_max: float = 0.0           # meeting room
    nx: int = 101
    t_final: float = 2.0
    nt: int = 4001
    # dynamics
    sigma: float = 0.3           # volatility
    max_speed: float = 5.0       # speed limit |a| <= a_max (gradient clipping)
    # costs
    alpha: float = 2.0           # lateness with respect to the scheduled time
    beta: float = 2.0            # lateness with respect to the actual start
    gamma: float = 1.0           # waiting
    t_scheduled: float = 0.5
    quorum: float = 0.90
    terminal_penalty: float = 5.0  # extra cost for agents still out at t_final
    wall_slope: float = 0.1      # u_x = -wall_slope at the wall: a small push towards the room
    # initial density: Gaussian bump
    m0_center: float = -2.5
    m0_width: float = 0.5

    @property
    def nu(self) -> float:
        return self.sigma ** 2 / 2

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dt(self) -> float:
        return self.t_final / (self.nt - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def t(self) -> np.ndarray:
        return np.linspace(0.0, self.t_final, self.nt)


def arrival_cost(t, T, p: Params):
    """Cost c(t, T) paid by an agent arriving at time t when the meeting starts at T."""
    t = np.asarray(t, dtype=float)
    return (p.alpha * np.maximum(t - p.t_scheduled, 0.0)
            + p.beta * np.maximum(t - T, 0.0)
            + p.gamma * np.maximum(T - t, 0.0))


# ── Hamilton–Jacobi–Bellman (backward) ─────────────────────────────────────────

def _clipped_gradient(u: np.ndarray, p: Params) -> np.ndarray:
    """Central differences inside, one-sided at the wall, clipped at the speed limit."""
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:] - u[:-2]) / (2 * p.dx)
    ux[0] = (u[1] - u[0]) / p.dx
    return np.clip(ux, -p.max_speed, p.max_speed)


def solve_hjb(T: float, p: Params, scheme: str = "explicit") -> np.ndarray:
    """Value function u(t, x) on the grid, for a meeting starting at T."""
    nt, nx, dx, dt, nu = p.nt, p.nx, p.dx, p.dt, p.nu
    u = np.zeros((nt, nx))
    u[-1, :] = p.terminal_penalty + arrival_cost(p.t_final, T, p)
    u[:, -1] = arrival_cost(p.t, T, p)                    # Dirichlet at the room

    if scheme == "implicit":
        # (I - dt nu D2) u^n = u^{n+1} - dt H(u_x^{n+1}) inside;
        # row 0: u_0 - u_1 = wall_slope dx (wall), last row: u = c(t_n, T).
        main = np.full(nx, 1 + 2 * dt * nu / dx ** 2)
        off = np.full(nx - 1, -dt * nu / dx ** 2)
        A = sparse.diags([off, main, off], [-1, 0, 1], format="lil")
        A[0, :] = 0
        A[0, 0], A[0, 1] = 1.0, -1.0
        A[-1, :] = 0
        A[-1, -1] = 1.0
        solve = splu(A.tocsc()).solve

    for n in range(nt - 2, -1, -1):
        u_next = u[n + 1]
        ux = _clipped_gradient(u_next, p)
        if scheme == "explicit":
            uxx = np.zeros(nx)
            uxx[1:-1] = (u_next[2:] - 2 * u_next[1:-1] + u_next[:-2]) / dx ** 2
            u[n, 1:-1] = u_next[1:-1] - dt * (0.5 * ux[1:-1] ** 2 - nu * uxx[1:-1])
            u[n, 0] = u[n, 1] + p.wall_slope * dx
        elif scheme == "implicit":
            b = u_next - dt * 0.5 * ux ** 2
            b[0] = p.wall_slope * dx
            b[-1] = u[n, -1]
            u[n] = solve(b)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    return u


def velocity(u: np.ndarray, p: Params) -> np.ndarray:
    """Optimal drift a* = -u_x, clipped at the speed limit."""
    v = np.zeros_like(u)
    v[:, 1:-1] = -(u[:, 2:] - u[:, :-2]) / (2 * p.dx)
    v[:, 0] = -(u[:, 1] - u[:, 0]) / p.dx
    v = np.clip(v, -p.max_speed, p.max_speed)
    v[:, -1] = v[:, -2]
    return v


# ── Fokker–Planck (forward) ────────────────────────────────────────────────────

def initial_density(p: Params) -> np.ndarray:
    m0 = np.exp(-((p.x - p.m0_center) ** 2) / (2 * p.m0_width ** 2))
    m0[-1] = 0.0                                          # nobody starts in the room
    return m0 / (m0.sum() * p.dx)


def solve_fp(v: np.ndarray, p: Params, scheme: str = "explicit"):
    """Density m(t, x) and cumulative arrivals for the drift v.

    Conservative finite volumes on the cells 0..nx-2 (cell nx-1 is the room,
    where m = 0): upwind advective fluxes and centred diffusive fluxes at the
    faces between cells, zero flux through the wall.  Mass therefore only
    leaves through the room, and the arrivals are exactly 1 - mass.
    """
    nt, nx, dx, dt, nu = p.nt, p.nx, p.dx, p.dt, p.nu
    m = np.zeros((nt, nx))
    m[0] = initial_density(p)
    arrivals = np.zeros(nt)

    if scheme == "implicit":
        # (I - dt nu L) m^{n+1} = m^n - dt/dx (advective flux balance), with L
        # the diffusion operator: zero flux at the wall, m = 0 in the room.
        k = nx - 1
        main = np.full(k, 2.0)
        main[0] = 1.0
        L = sparse.diags([np.ones(k - 1), -main, np.ones(k - 1)], [-1, 0, 1]) / dx ** 2
        solve = splu((sparse.eye(k) - dt * nu * L).tocsc()).solve
    elif scheme != "explicit":
        raise ValueError(f"unknown scheme {scheme!r}")

    for n in range(nt - 1):
        mc = m[n]
        vf = 0.5 * (v[n, :-1] + v[n, 1:])                  # velocity at the nx-1 faces
        flux = np.where(vf > 0, vf * mc[:-1], vf * mc[1:])  # upwind advection
        if scheme == "explicit":
            flux = flux - nu * (mc[1:] - mc[:-1]) / dx       # diffusion
        inflow = np.concatenate(([0.0], flux[:-1]))         # zero flux at the wall
        rhs = mc[:-1] - (dt / dx) * (flux - inflow)
        m[n + 1, :-1] = rhs if scheme == "explicit" else solve(rhs)
        arrivals[n + 1] = 1.0 - m[n + 1].sum() * dx
    return m, arrivals


# ── Fixed point on the starting time ───────────────────────────────────────────

@dataclass
class Result:
    T: float
    converged: bool
    history: list = field(default_factory=list)   # (guess, quorum time) per iteration
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    m: np.ndarray | None = None
    arrivals: np.ndarray | None = None


def quorum_time(arrivals: np.ndarray, p: Params) -> float:
    """First grid time at which the arrived mass reaches the quorum (t_final if never)."""
    reached = np.nonzero(arrivals >= p.quorum)[0]
    return float(p.t[reached[0]]) if len(reached) else p.t_final


def solve_meeting_time(p: Params | None = None, scheme: str = "explicit", T0: float = 1.2,
                       tol: float = 5e-3, max_iter: int = 30, damping: float = 0.5,
                       verbose: bool = False) -> Result:
    """Damped fixed point T <- (1 - damping) T + damping T_new."""
    p = p or Params(nt=DEFAULT_NT[scheme])
    res, T = Result(T=T0, converged=False), T0
    for it in range(max_iter):
        u = solve_hjb(T, p, scheme)
        v = velocity(u, p)
        m, arrivals = solve_fp(v, p, scheme)
        T_new = quorum_time(arrivals, p)
        res.history.append((T, T_new))
        if verbose:
            print(f"iter {it + 1:2d}: guess T = {T:.4f} -> quorum at {T_new:.4f}"
                  f"  (|diff| = {abs(T_new - T):.4f})")
        res.u, res.v, res.m, res.arrivals = u, v, m, arrivals
        if abs(T_new - T) < tol:
            res.T, res.converged = T, True
            return res
        T = (1 - damping) * T + damping * T_new
    res.T = T
    return res


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_density_and_arrivals(res: Result, p: Params, path: Path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im = ax1.imshow(res.m, aspect="auto", extent=[p.x_min, p.x_max, p.t_final, 0],
                    cmap="viridis")
    fig.colorbar(im, ax=ax1, label="Density")
    ax1.axhline(res.T, color="white", linestyle="--", alpha=0.5, label="Meeting start")
    ax1.set(title=f"Density of agents (meeting T = {res.T:.2f})",
            xlabel="Position (x)", ylabel="Time (t)")
    ax1.legend()
    ax2.plot(p.t, res.arrivals, linewidth=2, label="Proportion arrived")
    ax2.axhline(p.quorum, color="r", linestyle="--", label=f"{p.quorum:.0%} quorum")
    ax2.axvline(res.T, color="g", label=f"Start T = {res.T:.2f}")
    ax2.set(xlim=(0, p.t_final), ylim=(0, 1.05), title="Cumulative arrivals at the meeting",
            xlabel="Time", ylabel="Proportion")
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_surface(res: Result, p: Params, path: Path, every: int = 20):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, every * (p.nt - 1) // 4000)
    T_mesh, X_mesh = np.meshgrid(p.t[::step], p.x)
    surf = ax.plot_surface(X_mesh, T_mesh, res.m[::step].T, cmap="viridis",
                           edgecolor="none", alpha=0.9)
    ax.set(title=f"Density evolution (meeting T = {res.T:.2f})",
           xlabel="Position (x)", ylabel="Time (t)", zlabel="Density")
    ax.view_init(elev=30, azim=-120)
    fig.colorbar(surf, shrink=0.5, aspect=5, label="Density")
    fig.savefig(path, dpi=150)
    return fig


def animate(res: Result, p: Params, path: Path, frames: int = 200):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(xlim=(p.x_min, p.x_max), ylim=(0, res.m.max() * 1.1), xlabel="Position",
           ylabel="Density", title="Crowd movement")
    ax.grid(True)
    ax.axvline(0, color="red", label="Meeting room")
    line, = ax.plot([], [], lw=2, color="blue", label="Density")
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend()

    def update(n):
        line.set_data(p.x, res.m[n])
        started = p.t[n] >= res.T
        text.set_text(f"Time: {p.t[n]:.2f} "
                      f"({'MEETING STARTED' if started else 'meeting not started'})")
        text.set_color("red" if started else "black")
        return line, text

    idx = np.linspace(0, p.nt - 1, frames).astype(int)
    FuncAnimation(fig, update, frames=idx, blit=True).save(path, writer=PillowWriter(fps=30))
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scheme", choices=SCHEMES, default="explicit")
    ap.add_argument("--nt", type=int, default=None,
                    help="time steps (default: 4001 explicit, 401 implicit)")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--no-animation", action="store_true", help="skip the GIF")
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Params(nt=args.nt or DEFAULT_NT[args.scheme])
    print(f"{args.scheme} scheme, nx = {p.nx}, nt = {p.nt}")
    res = solve_meeting_time(p, args.scheme, verbose=True)
    print(f"{'converged' if res.converged else 'NOT converged'}: T* = {res.T:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_density_and_arrivals(res, p, args.out / "meeting_time.png")
    plot_surface(res, p, args.out / "meeting_time_3d.png")
    if not args.no_animation:
        animate(res, p, args.out / "meeting_crowd.gif")
    if args.show:
        plt.show()
    return res


if __name__ == "__main__":
    main()
