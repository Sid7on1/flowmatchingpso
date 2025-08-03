import numpy as np
from scipy import stats
import logging
import logging.config
from typing import Dict, List, Tuple
from flow_matching_pso.config import Config
from flow_matching_pso.exceptions import EvaluationError
from flow_matching_pso.models import FlowMatchingModel, ParticleSwarmOptimizationModel
from flow_matching_pso.utils import load_config, setup_logging

# Set up logging
logging.config.dictConfig(load_config('logging'))
logger = logging.getLogger(__name__)

class Evaluation:
    """
    Defines the evaluation metrics and procedures for the hybrid algorithm.
    """

    def __init__(self, config: Config):
        """
        Initializes the Evaluation object with the given configuration.

        Args:
            config (Config): The configuration object.
        """
        self.config = config
        self.flow_matching_model = FlowMatchingModel(config)
        self.particle_swarm_optimization_model = ParticleSwarmOptimizationModel(config)

    def calculate_metrics(self, results: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculates the evaluation metrics for the given results.

        Args:
            results (Dict[str, List[float]]): The results dictionary.

        Returns:
            Dict[str, float]: The evaluation metrics dictionary.
        """
        try:
            # Calculate the mean and standard deviation of the results
            mean = np.mean(results['values'])
            std_dev = np.std(results['values'])

            # Calculate the median and interquartile range of the results
            median = np.median(results['values'])
            iqr = stats.iqr(results['values'])

            # Calculate the number of outliers
            outliers = np.sum(np.abs(results['values'] - mean) > 1.5 * std_dev)

            # Calculate the percentage of outliers
            outlier_percentage = (outliers / len(results['values'])) * 100

            # Create the evaluation metrics dictionary
            metrics = {
                'mean': mean,
                'std_dev': std_dev,
                'median': median,
                'iqr': iqr,
                'outlier_percentage': outlier_percentage
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise EvaluationError("Error calculating metrics")

    def compare_results(self, results1: Dict[str, List[float]], results2: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compares the two sets of results and returns the evaluation metrics.

        Args:
            results1 (Dict[str, List[float]]): The first set of results.
            results2 (Dict[str[, List[float]]): The second set of results.

        Returns:
            Dict[str, float]: The evaluation metrics dictionary.
        """
        try:
            # Calculate the mean and standard deviation of the results
            mean1 = np.mean(results1['values'])
            std_dev1 = np.std(results1['values'])
            mean2 = np.mean(results2['values'])
            std_dev2 = np.std(results2['values'])

            # Calculate the median and interquartile range of the results
            median1 = np.median(results1['values'])
            iqr1 = stats.iqr(results1['values'])
            median2 = np.median(results2['values'])
            iqr2 = stats.iqr(results2['values'])

            # Calculate the number of outliers
            outliers1 = np.sum(np.abs(results1['values'] - mean1) > 1.5 * std_dev1)
            outliers2 = np.sum(np.abs(results2['values'] - mean2) > 1.5 * std_dev2)

            # Calculate the percentage of outliers
            outlier_percentage1 = (outliers1 / len(results1['values'])) * 100
            outlier_percentage2 = (outliers2 / len(results2['values'])) * 100

            # Create the evaluation metrics dictionary
            metrics = {
                'mean_diff': np.abs(mean1 - mean2),
                'std_dev_diff': np.abs(std_dev1 - std_dev2),
                'median_diff': np.abs(median1 - median2),
                'iqr_diff': np.abs(iqr1 - iqr2),
                'outlier_percentage_diff': np.abs(outlier_percentage1 - outlier_percentage2)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error comparing results: {str(e)}")
            raise EvaluationError("Error comparing results")

def main():
    # Load the configuration
    config = load_config('evaluation')

    # Set up the logging
    setup_logging(config)

    # Create the evaluation object
    evaluation = Evaluation(config)

    # Simulate some results
    results1 = {
        'values': np.random.rand(100)
    }
    results2 = {
        'values': np.random.rand(100)
    }

    # Calculate the metrics
    metrics = evaluation.calculate_metrics(results1)
    logger.info(f"Metrics: {metrics}")

    # Compare the results
    compared_metrics = evaluation.compare_results(results1, results2)
    logger.info(f"Compared Metrics: {compared_metrics}")

if __name__ == "__main__":
    main()