import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Normal
from flow_matching import FlowMatching
from particle_swarm_optimization import ParticleSwarmOptimization
from ordinary_differential_equations import ODESolver
from evaluation import Evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Main:
    """
    Main class for the FlowMatchingPSO project.

    ...

    Attributes
    ----------
    config : dict
        Configuration settings for the project.
    device : torch.device
        Torch device to use for computations.
    flow_matching : FlowMatching
        Instance of the FlowMatching class.
    pso : ParticleSwarmOptimization
        Instance of the ParticleSwarmOptimization class.
    ode_solver : ODESolver
        Instance of the ODESolver class for solving ordinary differential equations.
    eval_module : Evaluation
        Instance of the Evaluation class for evaluating results.

    Methods
    -------
    run_hybrid_algorithm(data, target_distribution):
        Runs the hybrid algorithm combining Flow Matching and PSO.
    evaluate_results(data, target_distribution):
        Evaluates the results of the hybrid algorithm.
    """

    def __init__(self, config):
        """
        Initializes the Main class with the project configuration.

        Parameters
        ----------
        config : dict
            Configuration settings for the project.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow_matching = FlowMatching(config)
        self.pso = ParticleSwarmOptimization(config)
        self.ode_solver = ODESolver(method=config['ode_solver']['method'])
        self.eval_module = Evaluation()

    def run_hybrid_algorithm(self, data, target_distribution):
        """
        Runs the hybrid algorithm combining Flow Matching and PSO.

        Parameters
        ----------
        data : torch.Tensor
            Input data samples.
        target_distribution : torch.distributions.Distribution
            Target distribution to match.

        Returns
        -------
        dict
            A dictionary containing the optimized flow matching parameters and
            the final particle positions and velocities from PSO.
        """
        logging.info('Running hybrid algorithm...')

        # Set device
        data = data.to(self.device)
        target_distribution = target_distribution.to(self.device)

        # Initialize flow matching
        self.flow_matching.initialize(data, target_distribution)

        # Get initial particle positions from flow matching
        particle_positions = self.flow_matching.get_particle_positions()

        # Initialize PSO
        self.pso.initialize(particle_positions, target_distribution)

        # Main optimization loop
        logging.info('Starting optimization loop...')
        velocity_threshold = self.config['velocity_threshold']
        max_iter = self.config['max_iter']
        for iteration in range(max_iter):
            # Update particle velocities and positions using PSO
            self.pso.update_velocities()
            self.pso.update_positions()

            # Compute current particle velocities
            velocities = self.pso.get_velocities()

            # Check velocity threshold condition
            vel_norms = torch.norm(velocities, p=2, dim=1)
            if torch.all(vel_norms < velocity_threshold):
                logging.info(f'Velocity threshold reached at iteration {iteration}.')
                break

            # Update flow matching parameters using current particle positions
            self.flow_matching.update_parameters(self.pso.get_positions())

        # Get final particle positions and velocities
        final_positions = self.pso.get_positions()
        final_velocities = self.pso.get_velocities()

        # Get optimized flow matching parameters
        optimized_params = self.flow_matching.get_parameters()

        logging.info('Optimization loop completed.')

        # Return optimized flow matching parameters and final PSO particles
        return {'flow_matching_params': optimized_params,
                'final_positions': final_positions,
                'final_velocities': final_velocities}

    def evaluate_results(self, data, target_distribution):
        """
        Evaluates the results of the hybrid algorithm.

        Parameters
        ----------
        data : torch.Tensor
            Input data samples.
        target_distribution : torch.distributions.Distribution
            Target distribution to match.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics for the optimized flow
            matching and PSO results.
        """
        logging.info('Evaluating results...')

        # Set device
        data = data.to(self.device)
        target_distribution = target_distribution.to(self.device)

        # Get optimized flow matching parameters
        optimized_params = self.flow_matching.get_parameters()

        # Evaluate flow matching results
        fm_metrics = self.eval_module.evaluate_flow_matching(data, target_distribution, optimized_params)

        # Initialize PSO with final particle positions from hybrid algorithm
        final_positions = self.pso.get_positions()
        self.pso.initialize(final_positions, target_distribution)

        # Evaluate PSO results
        pso_metrics = self.eval_module.evaluate_pso(final_positions, target_distribution)

        logging.info('Evaluation completed.')

        # Return evaluation metrics
        return {'flow_matching_metrics': fm_metrics, 'pso_metrics': pso_metrics}

def main():
    # Load configuration
    try:
        config = load_config('config.yaml')
    except FileNotFoundError:
        logging.error('Configuration file not found.')
        return

    # Set random seeds for reproducibility
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Create Main instance
    main_module = Main(config)

    # Load data and target distribution
    try:
        data = load_data(config['data_path'])
        target_distribution = load_target_distribution(config['target_dist_params'])
    except FileNotFoundError as e:
        logging.error(f'Data or target distribution file not found: {e}')
        return

    # Run hybrid algorithm
    result = main_module.run_hybrid_algorithm(data, target_distribution)

    # Evaluate results
    eval_result = main_module.evaluate_results(data, target_distribution)

    # Save results
    save_results(result, eval_result, config['results_path'])

    logging.info('All tasks completed successfully.')

if __name__ == '__main__':
    main()