import os

import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# --- 1. CONFIGURATION & PARAMETERS ---
# Grid parameters
X_min = -5.0       # Starting furthest distance
X_max = 0.0        # Meeting location
Nx = 101           # Space steps
dx = (X_max - X_min) / (Nx - 1)
x_grid = np.linspace(X_min, X_max, Nx)

T_final = 2.0      # Max time horizon
Nt = 4001          # High resolution for stability
dt = T_final / (Nt - 1)
t_grid = np.linspace(0, T_final, Nt)

# Physics parameters
sigma = 0.3        # Volatility
nu = (sigma**2) / 2 
MAX_SPEED = 5.0    # <--- STABILITY FIX: Speed limit for agents

# Cost parameters
alpha = 2.0   # Early penalty
beta = 2.0    # Late penalty
gamma = 1.0   # Waiting penalty
target_percent = 0.90 

# --- 2. COST FUNCTION ---
def arrival_cost(t, T_meeting):
    t_scheduled = 0.5
    # If t > T_meeting, you are late relative to the start (beta)
    # If t < T_meeting, you wait (gamma)
    # If t > t_scheduled, you are late on schedule (alpha)
    
    c1 = alpha * max(t - t_scheduled, 0)
    c2 = beta * max(t - T_meeting, 0)
    c3 = gamma * max(T_meeting - t, 0)
    return c1 + c2 + c3

# --- 3. SOLVERS ---

def solve_hjb(T_meeting):
    """ 
    Backward HJB Solver with Gradient Clipping 
    Prevents overflow by limiting maximum agent speed.
    """
    u = np.zeros((Nt, Nx))

    # Terminal condition (Time T_final): High cost if you haven't arrived
    u[-1, :] = 5.0 + arrival_cost(T_final, T_meeting)
    
    # Boundary condition at x=0 (Arrival)
    for n in range(Nt):
        u[n, -1] = arrival_cost(t_grid[n], T_meeting)
        
    # Backward Loop
    for n in range(Nt - 2, -1, -1):
        u_next = u[n+1, :]
        
        # 1. Calculate derivatives (Central Difference)
        ux = np.zeros(Nx)
        ux[1:-1] = (u_next[2:] - u_next[:-2]) / (2*dx)
        ux[0] = (u_next[1] - u_next[0]) / dx # One-sided left
        
        # --- STABILITY FIX: CLAMP THE GRADIENT ---
        # This effectively limits the max speed agents can choose
        ux = np.clip(ux, -MAX_SPEED, MAX_SPEED)
        
        # 2. Second derivative (Laplacian)
        uxx = np.zeros(Nx)
        uxx[1:-1] = (u_next[2:] - 2*u_next[1:-1] + u_next[:-2]) / (dx**2)
        uxx[0] = uxx[1] # Extrapolate boundary
        
        # 3. Explicit Update
        # PDE: u_t + min_a [ a*u_x + 0.5*a^2 ] + nu*u_xx = 0
        # Opt a = -u_x
        # PDE: u_t = 0.5*(u_x)^2 - nu*u_xx
        
        d_u = 0.5 * (ux**2) - nu * uxx
        
        # Update interior points
        u[n, 1:-1] = u_next[1:-1] - dt * d_u[1:-1]
        
        # Left Boundary (Soft Wall / Neumann)
        u[n, 0] = u[n, 1] + dx * 0.1 
        
    # Calculate Velocity: v = - gradient(u)
    v = np.zeros((Nt, Nx))
    for n in range(Nt):
        raw_v_interior = -(u[n, 2:] - u[n, :-2]) / (2*dx)
        v[n, 1:-1] = np.clip(raw_v_interior, -MAX_SPEED, MAX_SPEED)
        
        # Boundary velocities
        v[n, 0] = np.clip(-(u[n, 1] - u[n, 0]) / dx, -MAX_SPEED, MAX_SPEED)
        v[n, -1] = v[n, -2] # Copy neighbor
        
    return u, v

def solve_fp(v):
    """ Forward Fokker-Planck Solver with Flux Conservation """
    m = np.zeros((Nt, Nx))
    
    # Initial Condition: Gaussian bump
    center = -2.5
    width = 0.5
    m[0, :] = np.exp(-((x_grid - center)**2) / (2*width**2))
    m[0, :] /= (np.sum(m[0, :]) * dx)
    
    cumulative_arrivals = np.zeros(Nt)
    
    for n in range(Nt - 1):
        m_curr = m[n, :]
        
        # 1. Diffusion Flux at interfaces (Shape: Nx-1)
        m_right = m_curr[1:]
        m_left  = m_curr[:-1]
        dm_dx_faces = (m_right - m_left) / dx
        diff_flux = -nu * dm_dx_faces
        
        # 2. Advection Flux at interfaces (Shape: Nx-1)
        # Interpolate velocity to faces
        v_current = v[n, :]
        v_faces = 0.5 * (v_current[:-1] + v_current[1:])
        
        # Upwind scheme for advection
        adv_flux = np.where(v_faces > 0, 
                            v_faces * m_left, 
                            v_faces * m_right)
                            
        # 3. Total Flux
        Total_Flux = adv_flux + diff_flux
        
        # 4. Update Density (Interior points 1..Nx-2)
        # Flux into cell i comes from i-1, Flux out goes to i
        m[n+1, 1:-1] = m_curr[1:-1] - (dt/dx) * (Total_Flux[1:] - Total_Flux[:-1])
        
        # Boundaries
        m[n+1, 0] = m[n+1, 1]     # Reflective left
        m[n+1, -1] = 0.0          # Absorbing right (Meeting point)
        
        # Track Arrivals (Mass that left the grid)
        # We integrate the flux leaving the rightmost interface
        # Flux leaving right boundary ~ v[-1]*m[-1]. 
        # But simpler: Conservation of mass. 
        total_mass = np.sum(m[n+1, :]) * dx
        # Any mass missing is mass that arrived
        cumulative_arrivals[n+1] = max(0, 1.0 - total_mass)

    return m, cumulative_arrivals

# --- 4. MAIN ITERATION LOOP ---
T_star = 1.2
print(f"Initial Guess T*: {T_star}")

tolerance = 5e-3 # Slightly looser tolerance is fine for plotting
max_iters = 30

for i in range(max_iters):
    u, v = solve_hjb(T_star)
    m, arrivals = solve_fp(v)
    
    # Find new meeting time
    # Check if we ever crossed the target
    if arrivals[-1] < target_percent:
        print(f"Iter {i+1}: Quorum not reached (Max arrivals: {arrivals[-1]:.2f})")
        T_new = T_final # Push time later
    else:
        # Find first time t where arrivals >= target
        idx = np.argmax(arrivals >= target_percent)
        T_new = t_grid[idx]
    
    diff = abs(T_new - T_star)
    print(f"Iter {i+1}: Guess T*={T_star:.4f} -> Result T={T_new:.4f} (Diff: {diff:.4f})")
    
    if diff < tolerance:
        print("Converged!")
        break
        
    # Weighted update
    T_star = 0.5 * T_star + 0.5 * T_new

# --- 5. VISUALIZATION ---
plt.figure(figsize=(14, 6))

# Heatmap
plt.subplot(1, 2, 1)
# Note: extent=[left, right, bottom, top]
plt.imshow(m, aspect='auto', extent=[X_min, X_max, T_final, 0], cmap='viridis')
plt.colorbar(label='Density')
plt.title(f"Density of Agents (Meeting T={T_star:.2f})")
plt.xlabel("Position (x)")
plt.ylabel("Time (t)")
plt.axhline(T_star, color='white', linestyle='--', alpha=0.5, label='Meeting Start')
plt.legend()

# Arrivals
plt.subplot(1, 2, 2)
plt.plot(t_grid, arrivals, linewidth=2, label='Prop. Arrived')
plt.axhline(target_percent, color='r', linestyle='--', label=f'{target_percent*100}% Quorum')
plt.axvline(T_star, color='g', linestyle='-', label=f'Calculated Start T={T_star:.2f}')
plt.xlim(0, T_final)
plt.ylim(0, 1.05)
plt.title("Cumulative Arrivals at Meeting")
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/meeting_time.png", dpi=150)
plt.show()


# Visualization extras 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

# Ensure we have the necessary data from the simulation
# (m, v, x_grid, t_grid, T_star, X_min, X_max, T_final)

print("Generating Advanced Visualizations...")

# --- 1. 3D SURFACE PLOT ---
# Shows the density evolution as a landscape
fig_3d = plt.figure(figsize=(12, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Create a meshgrid for plotting (downsample for performance if Nt is huge)
# We plot every 20th time step to keep the graph fast
step_skip = 20
T_mesh, X_mesh = np.meshgrid(t_grid[::step_skip], x_grid)
M_mesh = m[::step_skip, :].T  # Transpose to match meshgrid (x, t)

surf = ax_3d.plot_surface(X_mesh, T_mesh, M_mesh, cmap='viridis', 
                          edgecolor='none', alpha=0.9)

ax_3d.set_title(f"3D Density Evolution (Meeting T={T_star:.2f})")
ax_3d.set_xlabel("Position (x)")
ax_3d.set_ylabel("Time (t)")
ax_3d.set_zlabel("Density")
ax_3d.view_init(elev=30, azim=-120) # Adjust camera angle
fig_3d.colorbar(surf, shrink=0.5, aspect=5, label='Density')
fig_3d.savefig("figures/meeting_time_3d.png", dpi=150)
plt.show()


# --- 2. ANIMATION ---
# Creates a movie of the density profile moving towards the door
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
ax_anim.set_xlim(X_min, X_max)
ax_anim.set_ylim(0, np.max(m) * 1.1)
ax_anim.set_xlabel("Position")
ax_anim.set_ylabel("Density")
ax_anim.set_title("Crowd Movement Animation")
ax_anim.grid(True)

# Static lines
ax_anim.axvline(0, color='red', label='Meeting Room')
line, = ax_anim.plot([], [], lw=2, color='blue', label='Density')
time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    # 'frame' is the index in t_grid
    current_t = t_grid[frame]
    current_density = m[frame, :]
    
    line.set_data(x_grid, current_density)
    
    # Change title/text based on status
    if current_t < T_star:
        status = "Meeting hasn't started"
        color = "black"
    else:
        status = "MEETING STARTED"
        color = "red"
        
    time_text.set_text(f"Time: {current_t:.2f} ({status})")
    time_text.set_color(color)
    return line, time_text

# Downsample frames for smoother/smaller GIF (e.g., every 10th frame)
frames = range(0, Nt, 20) 
ani = FuncAnimation(fig_anim, update, frames=frames, init_func=init, blit=True)

# To save the animation, you need pillow (for gif) or ffmpeg (for mp4)
ani.save("figures/meeting_crowd.gif", writer=PillowWriter(fps=30))

plt.legend()
plt.show()