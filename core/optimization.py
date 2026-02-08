"""
Multistage Optimization Module
Implements iterative optimization for bandwidth enhancement
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from .rf_calculations import SubstrateProperties, DesignSpecs
from .simulation import BalunSimulator, SimulationConfig


@dataclass
class OptimizationBounds:
    """Parameter bounds for optimization"""
    line_length_min: float = 0.8  # Multiplier on quarter-wave
    line_length_max: float = 1.2
    line_width_min: float = 0.8  # Multiplier on calculated width
    line_width_max: float = 1.2
    stage_offset_min: float = -0.1  # Frequency offset multiplier
    stage_offset_max: float = 0.1


@dataclass
class OptimizationResult:
    """Result of optimization run"""
    success: bool
    iterations: int
    best_cost: float
    best_params: Dict
    history: List[float] = field(default_factory=list)
    final_metrics: Dict = field(default_factory=dict)


class MultistageOptimizer:
    """
    Optimizer for multistage branchline balun
    
    Uses iterative tuning to optimize:
    - Line lengths
    - Line widths
    - Stage frequency offsets
    
    Goals:
    - Maximize bandwidth
    - Minimize return loss
    - Maintain phase balance
    """
    
    def __init__(self, substrate: SubstrateProperties, specs: DesignSpecs):
        self.substrate = substrate
        self.specs = specs
        self.simulator = BalunSimulator(substrate, specs)
        self.bounds = OptimizationBounds()
    
    def cost_function(self, params: Dict, weights: Dict = None) -> float:
        """
        Calculate cost for current parameters
        
        Lower cost = better design
        
        Args:
            params: Current design parameters
            weights: Weights for different objectives
        
        Returns:
            Cost value (lower is better)
        """
        if weights is None:
            weights = {
                'bandwidth': 1.0,
                'return_loss': 0.5,
                'phase_balance': 0.3,
                'amplitude_balance': 0.2
            }
        
        # Run simulation with current parameters
        num_stages = params.get('num_stages', 2)
        sim_result = self.simulator.simulate_multistage_balun(num_stages)
        metrics = self.simulator.calculate_performance_metrics(sim_result)
        
        # Calculate individual costs
        # Bandwidth: want to maximize (so negate)
        target_bw = self.specs.bandwidth_percent
        bw_cost = max(0, target_bw - metrics['bandwidth_percent']) / target_bw
        
        # Return loss: want < -10dB
        rl_cost = max(0, metrics['min_return_loss_db'] + 10) / 10
        
        # Phase balance: want ~180° (deviation < 5°)
        phase_cost = metrics['max_phase_imbalance_deg'] / 10
        
        # Amplitude balance: want < 1dB
        amp_cost = metrics['max_amplitude_imbalance_db']
        
        # Weighted sum
        total_cost = (
            weights['bandwidth'] * bw_cost +
            weights['return_loss'] * rl_cost +
            weights['phase_balance'] * phase_cost +
            weights['amplitude_balance'] * amp_cost
        )
        
        return total_cost
    
    def optimize_simple(self, max_iterations: int = 50, 
                        num_stages: int = 2,
                        verbose: bool = False) -> OptimizationResult:
        """
        Simple iterative optimization
        
        This is a simplified optimizer that tests discrete parameter variations
        and selects the best configuration.
        
        Args:
            max_iterations: Maximum optimization iterations
            num_stages: Number of balun stages
            verbose: Print progress
        
        Returns:
            OptimizationResult with best parameters
        """
        # Initial parameters
        best_params = {
            'num_stages': num_stages,
            'length_factor': 1.0,
            'width_factor': 1.0,
            'stage_offsets': [0.0] * num_stages
        }
        
        best_cost = self.cost_function(best_params)
        cost_history = [best_cost]
        
        # Grid search over parameter space
        length_factors = np.linspace(0.9, 1.1, 5)
        width_factors = np.linspace(0.9, 1.1, 5)
        
        iteration = 0
        
        for lf in length_factors:
            for wf in width_factors:
                iteration += 1
                if iteration > max_iterations:
                    break
                
                test_params = {
                    'num_stages': num_stages,
                    'length_factor': lf,
                    'width_factor': wf,
                    'stage_offsets': [0.0] * num_stages
                }
                
                cost = self.cost_function(test_params)
                cost_history.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = test_params.copy()
                    
                    if verbose:
                        print(f"Iteration {iteration}: New best cost = {cost:.4f}")
        
        # Get final metrics
        final_sim = self.simulator.simulate_multistage_balun(num_stages)
        final_metrics = self.simulator.calculate_performance_metrics(final_sim)
        
        return OptimizationResult(
            success=True,
            iterations=iteration,
            best_cost=best_cost,
            best_params=best_params,
            history=cost_history,
            final_metrics=final_metrics
        )
    
    def sensitivity_analysis(self, num_stages: int = 2) -> Dict[str, Dict]:
        """
        Analyze sensitivity of performance to parameter variations
        
        Args:
            num_stages: Number of stages to analyze
        
        Returns:
            Dictionary with sensitivity data for each parameter
        """
        base_params = {
            'num_stages': num_stages,
            'length_factor': 1.0,
            'width_factor': 1.0
        }
        
        base_cost = self.cost_function(base_params)
        
        # Test parameter variations
        variations = np.linspace(-0.1, 0.1, 11)
        
        results = {}
        
        # Length sensitivity
        length_costs = []
        for v in variations:
            params = base_params.copy()
            params['length_factor'] = 1.0 + v
            length_costs.append(self.cost_function(params))
        
        results['length_factor'] = {
            'variations': variations.tolist(),
            'costs': length_costs,
            'sensitivity': (max(length_costs) - min(length_costs)) / 0.2
        }
        
        # Width sensitivity
        width_costs = []
        for v in variations:
            params = base_params.copy()
            params['width_factor'] = 1.0 + v
            width_costs.append(self.cost_function(params))
        
        results['width_factor'] = {
            'variations': variations.tolist(),
            'costs': width_costs,
            'sensitivity': (max(width_costs) - min(width_costs)) / 0.2
        }
        
        return results
    
    def get_optimization_summary(self, result: OptimizationResult) -> str:
        """Generate text summary of optimization result"""
        lines = [
            "Optimization Summary",
            "=" * 40,
            f"Status: {'Success' if result.success else 'Failed'}",
            f"Iterations: {result.iterations}",
            f"Final Cost: {result.best_cost:.4f}",
            "",
            "Optimized Parameters:",
            f"  Number of Stages: {result.best_params['num_stages']}",
            f"  Length Factor: {result.best_params['length_factor']:.3f}",
            f"  Width Factor: {result.best_params['width_factor']:.3f}",
            "",
            "Performance Metrics:",
            f"  Bandwidth: {result.final_metrics.get('bandwidth_percent', 'N/A')}%",
            f"  Return Loss: {result.final_metrics.get('min_return_loss_db', 'N/A')} dB",
            f"  Phase Imbalance: {result.final_metrics.get('max_phase_imbalance_deg', 'N/A')}°",
            f"  Amplitude Imbalance: {result.final_metrics.get('max_amplitude_imbalance_db', 'N/A')} dB"
        ]
        
        return "\n".join(lines)


def quick_optimize(substrate: SubstrateProperties, specs: DesignSpecs, 
                   num_stages: int = 2) -> Tuple[Dict, Dict]:
    """
    Quick optimization function for use in Streamlit app
    
    Returns:
        Tuple of (optimized parameters, performance metrics)
    """
    optimizer = MultistageOptimizer(substrate, specs)
    result = optimizer.optimize_simple(max_iterations=25, num_stages=num_stages)
    
    return result.best_params, result.final_metrics
