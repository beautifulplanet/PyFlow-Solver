import sys
import numpy as np

class LiveLogger:
    """
    A logger to provide real-time, detailed feedback during a simulation.
    """
    def __init__(self, N, Re, dt, T, log_interval=100):
        self.N = N
        self.Re = Re
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)
        self.log_interval = log_interval
        self.last_len = 0

    def log_header(self):
        """Prints the header for the simulation log."""
        header = (
            f"Starting simulation:\n"
            f"  Grid size (N): {self.N}\n"
            f"  Reynolds (Re): {self.Re}\n"
            f"  Time step (dt): {self.dt}\n"
            f"  Total time (T): {self.T}\n"
            f"  Total steps: {self.nt}\n"
            f"  Logging every: {self.log_interval} steps\n"
            f"{'-'*60}"
        )
        print(header)

    def log_step(self, n, u, v, p, res=None):
        """
        Logs the state of a single simulation step, respecting the log_interval.
        
        Parameters:
        -----------
        n : int
            Current time step number
        u, v : ndarray
            Velocity components
        p : ndarray
            Pressure field
        res : dict, optional
            Dictionary containing residual information
        """
        if (n + 1) % self.log_interval != 0 and (n + 1) != self.nt:
            return

        percent = (n + 1) / self.nt * 100
        sim_time = (n + 1) * self.dt
        max_u = np.abs(u).max()
        max_v = np.abs(v).max()
        max_p = np.abs(p).max()
        
        log_line = (
            f"Progress: {percent:6.2f}% | Step: {n+1:6d}/{self.nt} | "
            f"Time: {sim_time:8.4f}s | "
            f"max|u|: {max_u:8.4f} | max|v|: {max_v:8.4f} | max|p|: {max_p:8.4f}"
        )
        
        if res is not None:
            # Add residuals information if provided
            log_line += (
                f" | u_res: {res.get('u_res', 0):8.2e} | "
                f"v_res: {res.get('v_res', 0):8.2e} | "
                f"cont_res: {res.get('cont_res', 0):8.2e}"
            )
        
        # Pad with spaces to clear the rest of the line and use carriage return
        padding = ' ' * max(0, self.last_len - len(log_line))
        sys.stdout.write(f"\r{log_line}{padding}")
        sys.stdout.flush()
        self.last_len = len(log_line)
    
    def log(self, t, step, u_res, v_res, cont_res):
        """
        Alternative log method for simpler residual logging.
        
        Parameters:
        -----------
        t : float
            Current simulation time
        step : int
            Current step number
        u_res, v_res, cont_res : float
            Residuals for u, v velocities and continuity
        """
        percent = t / self.T * 100
        
        log_line = (
            f"Progress: {percent:6.2f}% | Step: {step:6d}/{self.nt} | "
            f"Time: {t:8.4f}s | "
            f"u_res: {u_res:8.2e} | "
            f"v_res: {v_res:8.2e} | "
            f"cont_res: {cont_res:8.2e}"
        )
        
        # Pad with spaces to clear the rest of the line and use carriage return
        padding = ' ' * max(0, self.last_len - len(log_line))
        sys.stdout.write(f"\r{log_line}{padding}")
        sys.stdout.flush()
        self.last_len = len(log_line)

    def log_footer(self):
        """Prints a footer to conclude the log."""
        print(f"\n{'-'*60}\nSimulation finished.\n")
