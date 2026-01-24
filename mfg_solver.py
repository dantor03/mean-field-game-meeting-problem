import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MFGSolver1D:
    def __init__(self, T=1.0, Nt=50, x_min=-2.0, x_max=2.0, Nx=100, nu=0.1, interaction_strength=1.0):
        """
        1D Mean Field Game Solver using Finite Differences.
        Problem: Agents start at x ~ N(-1, 0.2) and want to go to x=1, 
        avoiding high density areas (Lasry-Lions style).
        """
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt
        self.x_min = x_min
        self.x_max = x_max
        self.Nx = Nx
        self.dx = (x_max - x_min) / Nx
        self.nu = nu # Viscosity (diffusion/noise)
        self.xi = interaction_strength # Cost of crowding
        
        # Grids
        self.x = np.linspace(x_min, x_max, Nx)
        self.t = np.linspace(0, T, Nt)
        
        # Variables
        self.u = np.zeros((Nt, Nx)) # Value function (HJB)
        self.m = np.zeros((Nt, Nx)) # Density (Fokker-Planck)

    def initialize_distribution(self):
        # Initial distribution m(0,x): Gaussian centered at -1
        m0 = np.exp(-(self.x + 1.0)**2 / (2 * 0.1**2))
        self.m[0, :] = m0 / (np.sum(m0) * self.dx) # Normalize

    def solve_hjb_backward(self, m_guess):
        """Solve Hamilton-Jacobi-Bellman backward in time"""
        # Terminal condition: Agents want to be at x=1
        # Cost is (x - 1)^2
        self.u[-1, :] = (self.x - 1.0)**2
        
        for n in range(self.Nt - 2, -1, -1):
            # Finite Difference Schemes
            u_xx = (np.roll(self.u[n+1], -1) - 2*self.u[n+1] + np.roll(self.u[n+1], 1)) / (self.dx**2)
            u_x = (np.roll(self.u[n+1], -1) - np.roll(self.u[n+1], 1)) / (2*self.dx)
            
            # Hamiltonian: H(p) = |p|^2 / 2  --> H*(p) = -|p|^2 / 2
            # Running cost includes interaction term F(m)
            interaction_cost = self.xi * m_guess[n+1]
            
            # Update u (Backward Euler explicit-ish step)
            self.u[n] = self.u[n+1] - self.dt * (
                self.nu * u_xx - 0.5 * (u_x**2) + interaction_cost
            )
            
            # Boundary conditions (Neumann)
            self.u[n, 0] = self.u[n, 1]
            self.u[n, -1] = self.u[n, -2]

    def solve_kfp_forward(self):
        """Solve Kolmogorov-Fokker-Planck forward in time"""
        for n in range(0, self.Nt - 1):
            # Optimal control alpha = -grad(u)
            u_x = (np.roll(self.u[n], -1) - np.roll(self.u[n], 1)) / (2*self.dx)
            drift = -u_x
            
            # FDM for KFP
            m_xx = (np.roll(self.m[n], -1) - 2*self.m[n] + np.roll(self.m[n], 1)) / (self.dx**2)
            
            # Upwind scheme for drift (advection)
            flux = drift * self.m[n]
            flux_x = (np.roll(flux, -1) - np.roll(flux, 1)) / (2*self.dx) # Simplified central
            
            self.m[n+1] = self.m[n] + self.dt * (self.nu * m_xx - flux_x)
            
            # Normalize to ensure mass conservation
            self.m[n+1] = np.maximum(self.m[n+1], 0) # Positivity
            self.m[n+1] /= (np.sum(self.m[n+1]) * self.dx)

    def train(self, iterations=50):
        print(f"Solving MFG System ({iterations} iterations)...")
        self.initialize_distribution()
        
        # Fixed Point Iteration
        for i in range(iterations):
            m_prev = self.m.copy()
            self.solve_hjb_backward(self.m)
            self.solve_kfp_forward()
            
            error = np.linalg.norm(self.m - m_prev)
            if i % 10 == 0:
                print(f"Iter {i}: Error = {error:.5f}")
            if error < 1e-4:
                print("Converged!")
                break
                
    def plot_results(self):
        plt.figure(figsize=(12, 5))
        
        # Plot Heatmap of density evolution
        plt.subplot(1, 2, 1)
        sns.heatmap(self.m, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("Evolution of Agent Density m(t,x)")
        plt.xlabel("Position (x)")
        plt.ylabel("Time (t)")
        
        # Plot snapshots
        plt.subplot(1, 2, 2)
        times = [0, int(self.Nt/2), self.Nt-1]
        for t_idx in times:
            plt.plot(self.x, self.m[t_idx, :], label=f"t={t_idx*self.dt:.1f}")
        plt.title("Density Profiles at snapshots")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("mfg_results.png")
        print("Results saved to mfg_results.png")

if __name__ == "__main__":
    solver = MFGSolver1D(Nx=100, Nt=200, interaction_strength=2.0)
    solver.train(iterations=100)
    solver.plot_results()
