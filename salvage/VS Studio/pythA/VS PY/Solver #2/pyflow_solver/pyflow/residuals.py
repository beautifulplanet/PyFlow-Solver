import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class ResidualTracker:
    """
    Class for tracking and visualizing residuals during CFD simulations.
    """
    
    def __init__(self):
        self.u_residuals = []
        self.v_residuals = []
        self.continuity_residuals = []
        self.iterations = []
        self.iteration_count = 0
    
    def compute_residuals(self, u: np.ndarray, v: np.ndarray, 
                          u_prev: np.ndarray, v_prev: np.ndarray,
                          dx: float, dy: float) -> Tuple[float, float, float]:
        """
        Compute normalized residuals for u, v, and continuity equations.
        
        Parameters:
        -----------
        u, v: Current velocity fields
        u_prev, v_prev: Previous iteration velocity fields
        dx, dy: Grid spacing
        
        Returns:
        --------
        u_res, v_res, cont_res: Residuals for u, v velocities and continuity
        """
        # Momentum residuals (L2 norm of change in velocity)
        u_res = np.sqrt(np.mean((u - u_prev)**2)) / (np.mean(np.abs(u)) + 1e-12)
        v_res = np.sqrt(np.mean((v - v_prev)**2)) / (np.mean(np.abs(v)) + 1e-12)
        
        # Continuity residual (L2 norm of divergence)
        div = np.zeros_like(u)
        N = u.shape[0]
        for j in range(1, N-1):
            for i in range(1, N-1):
                div[j, i] = (u[j, i+1] - u[j, i-1])/(2*dx) + (v[j+1, i] - v[j-1, i])/(2*dy)
        
        cont_res = np.sqrt(np.mean(div**2))
        
        return u_res, v_res, cont_res
    
    def add_residuals(self, u_res: float, v_res: float, cont_res: float):
        """
        Add residuals to the tracking lists.
        """
        self.iteration_count += 1
        self.iterations.append(self.iteration_count)
        self.u_residuals.append(u_res)
        self.v_residuals.append(v_res)
        self.continuity_residuals.append(cont_res)
    
    def plot_residuals(self, title: Optional[str] = None, 
                      save_path: Optional[str] = None):
        """
        Plot the residual history.
        
        Parameters:
        -----------
        title: Optional title for the plot
        save_path: Optional file path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.u_residuals, 'b-', label='U-momentum')
        plt.plot(self.iterations, self.v_residuals, 'r-', label='V-momentum')
        plt.plot(self.iterations, self.continuity_residuals, 'g-', label='Continuity')
        
        plt.xlabel('Iteration')
        plt.ylabel('Normalized Residual')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        
        if title:
            plt.title(title)
        else:
            plt.title('Convergence History')
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def clear(self):
        """
        Reset the residual tracking data.
        """
        self.u_residuals = []
        self.v_residuals = []
        self.continuity_residuals = []
        self.iterations = []
        self.iteration_count = 0
