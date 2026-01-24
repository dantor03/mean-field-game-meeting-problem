# Mean Field Games: The Meeting Problem

Implementation of a classic **Mean Field Game (MFG)** system using Finite Difference Methods. This project simulates a population of agents optimizing their trajectories to reach a target while minimizing congestion costs, based on the **Lasry-Lions** formalism.

![Results](mfg_results.png)
*(Run the script to generate this plot)*

## 📝 Mathematical Formulation

The system consists of two coupled Partial Differential Equations (PDEs):

1. **Hamilton-Jacobi-Bellman (HJB) Equation**:
   Describes the optimal value function $u(t,x)$ for a representative agent (backward in time).
   $$-\partial_t u - \nu \Delta u + H(\nabla u) = F(m(t,x))$$

2. **Fokker-Planck (KFP) Equation**:
   Describes the evolution of the population density $m(t,x)$ under the optimal control (forward in time).
   $$\partial_t m - \nu \Delta m + \text{div}(m \cdot \nabla_p H(\nabla u)) = 0$$

Where $F(m)$ represents the **interaction cost** (e.g., aversion to crowds). The system is solved using a Fixed Point Iteration method.

## 🛠 Technologies
- **Python**: Core logic
- **NumPy**: Matrix operations and Finite Difference grids
- **Matplotlib/Seaborn**: Visualization of density heatmaps

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USER/mean-field-game-meeting-problem.git](https://github.com/YOUR_USER/mean-field-game-meeting-problem.git)
