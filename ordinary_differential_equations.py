# ordinary_differential_equations.py

import logging
import torch
import torchdiffeq
from typing import Callable, Tuple, Optional
from scipy.integrate import solve_ivp
from numpy import inf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrdinaryDifferentialEquations:
    """
    Defines the ordinary differential equations used in the flow matching module.
    """

    def __init__(self, 
                 ode_func: Callable, 
                 t_span: Tuple[float, float], 
                 t_eval: Optional[Tuple[float, float]] = None, 
                 method: str = 'RK45', 
                 rtol: float = 1e-5, 
                 atol: float = 1e-8):
        """
        Initializes the OrdinaryDifferentialEquations class.

        Args:
        ode_func (Callable): The ordinary differential equation function.
        t_span (Tuple[float, float]): The time span for the ODE solver.
        t_eval (Optional[Tuple[float, float]], optional): The time points to evaluate the solution. Defaults to None.
        method (str, optional): The method to use for the ODE solver. Defaults to 'RK45'.
        rtol (float, optional): The relative tolerance for the ODE solver. Defaults to 1e-5.
        atol (float, optional): The absolute tolerance for the ODE solver. Defaults to 1e-8.
        """
        self.ode_func = ode_func
        self.t_span = t_span
        self.t_eval = t_eval
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def define_ode(self, 
                   x0: torch.Tensor, 
                   t: torch.Tensor) -> torch.Tensor:
        """
        Defines the ordinary differential equation.

        Args:
        x0 (torch.Tensor): The initial condition.
        t (torch.Tensor): The time points.

        Returns:
        torch.Tensor: The solution of the ODE.
        """
        try:
            # Define the ODE function
            def ode_func(t, x):
                return self.ode_func(t, x)

            # Solve the ODE using scipy's solve_ivp
            sol = solve_ivp(ode_func, self.t_span, x0, t_eval=t, method=self.method, rtol=self.rtol, atol=self.atol)

            # Return the solution
            return sol.y.T

        except Exception as e:
            # Log the error
            logger.error(f"Error defining ODE: {e}")
            raise

    def solve_ode(self, 
                  x0: torch.Tensor, 
                  t: torch.Tensor) -> torch.Tensor:
        """
        Solves the ordinary differential equation.

        Args:
        x0 (torch.Tensor): The initial condition.
        t (torch.Tensor): The time points.

        Returns:
        torch.Tensor: The solution of the ODE.
        """
        try:
            # Define the ODE function
            def ode_func(t, x):
                return self.ode_func(t, x)

            # Solve the ODE using torchdiffeq's odeint
            sol = torchdiffeq.odeint(ode_func, x0, t)

            # Return the solution
            return sol

        except Exception as e:
            # Log the error
            logger.error(f"Error solving ODE: {e}")
            raise


class FlowTheoryODE:
    """
    Defines the ordinary differential equation for the flow theory.
    """

    def __init__(self, 
                 alpha: float = 0.5, 
                 beta: float = 0.5, 
                 gamma: float = 0.5):
        """
        Initializes the FlowTheoryODE class.

        Args:
        alpha (float, optional): The alpha parameter. Defaults to 0.5.
        beta (float, optional): The beta parameter. Defaults to 0.5.
        gamma (float, optional): The gamma parameter. Defaults to 0.5.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, t, x):
        """
        Defines the ordinary differential equation.

        Args:
        t (torch.Tensor): The time points.
        x (torch.Tensor): The state variables.

        Returns:
        torch.Tensor: The derivative of the state variables.
        """
        dxdt = torch.zeros_like(x)
        dxdt[0] = self.alpha * x[0] + self.beta * x[1]
        dxdt[1] = self.gamma * x[0] + self.alpha * x[1]
        return dxdt


class VelocityThresholdODE:
    """
    Defines the ordinary differential equation for the velocity threshold.
    """

    def __init__(self, 
                 v_threshold: float = 0.5):
        """
        Initializes the VelocityThresholdODE class.

        Args:
        v_threshold (float, optional): The velocity threshold. Defaults to 0.5.
        """
        self.v_threshold = v_threshold

    def __call__(self, t, x):
        """
        Defines the ordinary differential equation.

        Args:
        t (torch.Tensor): The time points.
        x (torch.Tensor): The state variables.

        Returns:
        torch.Tensor: The derivative of the state variables.
        """
        dxdt = torch.zeros_like(x)
        dxdt[0] = self.v_threshold * x[0]
        return dxdt


def main():
    # Define the ODE function for the flow theory
    flow_theory_ode = FlowTheoryODE()

    # Define the ODE function for the velocity threshold
    velocity_threshold_ode = VelocityThresholdODE()

    # Define the time points
    t = torch.linspace(0, 10, 100)

    # Define the initial condition
    x0 = torch.tensor([1.0, 2.0])

    # Define the ODE solver
    ode_solver = OrdinaryDifferentialEquations(flow_theory_ode, (0, 10))

    # Solve the ODE
    sol = ode_solver.solve_ode(x0, t)

    # Print the solution
    print(sol)


if __name__ == "__main__":
    main()