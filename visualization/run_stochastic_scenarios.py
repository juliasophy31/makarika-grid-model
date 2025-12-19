#!/usr/bin/env python3
"""
Stochastic Simulation Runner

Run multiple stochastic scenarios for Monte Carlo analysis of control strategies.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
import sys
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network_manager import NetworkManager
from scenarios.scenario_manager import ScenarioManager
from simulation.simulator import GridSimulator
from utils.cost_calculator import CostCalculator
from utils.stochastic_generator import StochasticScenarioManager


class StochasticSimulationRunner:
    """Run Monte Carlo simulations with stochastic profiles"""
    
    def __init__(
        self,
        base_dir: Path,
        n_runs: int = 10,
        load_noise_std: float = 0.10,
        pv_noise_std: float = 0.20,
        seed: int = 42
    ):
        self.base_dir = Path(base_dir)
        self.n_runs = n_runs
        self.load_noise_std = load_noise_std
        self.pv_noise_std = pv_noise_std
        self.seed = seed
        
        # Setup paths
        self.data_dir = self.base_dir / "venv/data/"
        self.config_path = self.base_dir / "config/scenario_definitions.yaml"
        self.results_dir = self.base_dir / "results/stochastic"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing stochastic simulation framework...")
        self.network_mgr = NetworkManager(self.data_dir)
        self.network_mgr.build_base_network()
        
        self.scenario_mgr = ScenarioManager(self.config_path, self.network_mgr, self.data_dir)
        self.simulator = GridSimulator(self.network_mgr, self.scenario_mgr, self.data_dir)
        self.cost_calculator = CostCalculator(config_path=self.config_path)
        
        # Initialize stochastic manager
        self.stochastic_mgr = StochasticScenarioManager(
            scenario_manager=self.scenario_mgr,
            n_runs=n_runs,
            load_noise_std=load_noise_std,
            pv_noise_std=pv_noise_std,
            seed=seed
        )
    
    def run_stochastic_scenario(
        self,
        scenario_name: str,
        n_steps: int = 24
    ) -> Dict:
        """
        Run multiple stochastic realizations of a single scenario
        
        Returns:
        --------
        results : dict
            Contains statistics and individual run results
        """
        print(f"\n{'='*80}")
        print(f"STOCHASTIC ANALYSIS: {scenario_name}")
        print(f"{'='*80}")
        print(f"Runs: {self.n_runs}")
        print(f"Load noise std: {self.load_noise_std:.1%}")
        print(f"PV noise std: {self.pv_noise_std:.1%}")
        print(f"Seed: {self.seed}")
        print()
        
        # Storage for results
        run_results = []
        cost_results = []
        energy_results = []
        
        # Run all stochastic realizations
        for run_id in range(self.n_runs):
            print(f"\n{'-'*60}")
            print(f"STOCHASTIC RUN {run_id + 1}/{self.n_runs}")
            print(f"{'-'*60}")
            
            start_time = time.time()
            
            try:
                # Reset network
                self.network_mgr.reset_network()
                
                # Apply stochastic scenario
                self.stochastic_mgr.apply_stochastic_scenario(scenario_name, run_id)
                
                # Run simulation with custom output path
                run_output_dir = self.results_dir / f"{scenario_name}_run_{run_id:03d}"
                result = self.simulator.run_scenario_with_output(
                    scenario_name, 
                    n_steps=n_steps, 
                    output_dir=run_output_dir
                )
                
                if result is None:
                    print(f"❌ Run {run_id} failed")
                    continue
                
                # Calculate costs
                try:
                    cost_df, cost_summary = self.cost_calculator.analyze_scenario(result['output_path'])
                    cost_results.append({
                        'run_id': run_id,
                        'total_cost': cost_summary['total_cost'],
                        'peak_cost': cost_summary.get('peak_cost', 0),
                        'offpeak_cost': cost_summary.get('offpeak_cost', 0),
                        'export_revenue': cost_summary.get('export_revenue', 0)
                    })
                except Exception as e:
                    print(f"⚠️ Cost calculation failed for run {run_id}: {e}")
                    continue
                
                # Extract energy metrics
                data = result['data']
                energy_metrics = self._extract_energy_metrics(data, run_id)
                energy_results.append(energy_metrics)
                
                run_results.append({
                    'run_id': run_id,
                    'output_path': result['output_path'],
                    'success': True
                })
                
                elapsed = time.time() - start_time
                print(f"✅ Run {run_id} completed in {elapsed:.1f}s")
                print(f"   Cost: ${cost_summary['total_cost']:.2f}")
                
            except Exception as e:
                print(f"❌ Run {run_id} failed: {e}")
                run_results.append({
                    'run_id': run_id,
                    'success': False,
                    'error': str(e)
                })
                continue
        
        # Compile statistics
        statistics = self._compile_statistics(cost_results, energy_results)
        
        # Save results
        self._save_stochastic_results(scenario_name, {
            'scenario_name': scenario_name,
            'n_runs': self.n_runs,
            'parameters': {
                'load_noise_std': self.load_noise_std,
                'pv_noise_std': self.pv_noise_std,
                'seed': self.seed
            },
            'statistics': statistics,
            'cost_results': cost_results,
            'energy_results': energy_results,
            'run_results': run_results
        })
        
        # Print summary
        self._print_summary(scenario_name, statistics)
        
        return {
            'scenario_name': scenario_name,
            'statistics': statistics,
            'cost_results': cost_results,
            'energy_results': energy_results,
            'run_results': run_results
        }
    
    def run_multiple_scenarios(
        self,
        scenario_names: List[str],
        n_steps: int = 24
    ) -> Dict:
        """Run stochastic analysis for multiple scenarios"""
        print(f"\n{'='*80}")
        print(f"STOCHASTIC ANALYSIS - MULTIPLE SCENARIOS")
        print(f"{'='*80}")
        print(f"Scenarios: {len(scenario_names)}")
        print(f"Runs per scenario: {self.n_runs}")
        print(f"Total simulations: {len(scenario_names) * self.n_runs}")
        print()
        
        all_results = {}
        
        for scenario_name in scenario_names:
            result = self.run_stochastic_scenario(scenario_name, n_steps)
            all_results[scenario_name] = result
        
        # Create comparison summary
        comparison = self._create_comparison_summary(all_results)
        
        # Save comparison
        comparison_file = self.results_dir / "stochastic_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        comparison_csv = self.results_dir / "stochastic_comparison.csv"
        self._save_comparison_csv(comparison, comparison_csv)
        
        print(f"\n{'='*80}")
        print("STOCHASTIC COMPARISON SUMMARY")
        print(f"{'='*80}")
        self._print_comparison(comparison)
        
        return all_results
    
    def _extract_energy_metrics(self, data: pd.DataFrame, run_id: int) -> Dict:
        """Extract energy metrics from simulation data"""
        total_load = data['load'].sum()
        total_pv = data['pv_generation'].sum()
        battery_discharge = data['battery_power'][data['battery_power'] > 0].sum()
        battery_charge = abs(data['battery_power'][data['battery_power'] < 0].sum())
        grid_import = data['grid_power'][data['grid_power'] > 0].sum()
        grid_export = abs(data['grid_power'][data['grid_power'] < 0].sum())
        
        self_sufficiency = (total_pv / total_load * 100) if total_load > 0 else 0
        
        return {
            'run_id': run_id,
            'total_load': total_load,
            'total_pv': total_pv,
            'battery_charge': battery_charge,
            'battery_discharge': battery_discharge,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'self_sufficiency': self_sufficiency
        }
    
    def _compile_statistics(self, cost_results: List[Dict], energy_results: List[Dict]) -> Dict:
        """Compile statistical analysis of results"""
        if not cost_results or not energy_results:
            return {}
        
        # Cost statistics
        costs = [r['total_cost'] for r in cost_results]
        cost_stats = {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'min': np.min(costs),
            'max': np.max(costs),
            'p05': np.percentile(costs, 5),
            'p95': np.percentile(costs, 95),
            'cv': np.std(costs) / np.mean(costs) * 100  # Coefficient of variation
        }
        
        # Energy statistics
        energy_stats = {}
        energy_metrics = ['total_load', 'total_pv', 'battery_charge', 'battery_discharge', 
                         'grid_import', 'grid_export', 'self_sufficiency']
        
        for metric in energy_metrics:
            values = [r[metric] for r in energy_results]
            energy_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) * 100 if np.mean(values) > 0 else 0
            }
        
        return {
            'cost': cost_stats,
            'energy': energy_stats,
            'n_successful_runs': len(cost_results)
        }
    
    def _save_stochastic_results(self, scenario_name: str, results: Dict):
        """Save stochastic results to files"""
        # Save full results as JSON
        json_file = self.results_dir / f"{scenario_name}_stochastic_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save cost results as CSV
        if results['cost_results']:
            cost_df = pd.DataFrame(results['cost_results'])
            cost_csv = self.results_dir / f"{scenario_name}_stochastic_costs.csv"
            cost_df.to_csv(cost_csv, index=False)
        
        # Save energy results as CSV
        if results['energy_results']:
            energy_df = pd.DataFrame(results['energy_results'])
            energy_csv = self.results_dir / f"{scenario_name}_stochastic_energy.csv"
            energy_df.to_csv(energy_csv, index=False)
        
        print(f"📁 Results saved to: {self.results_dir}")
    
    def _create_comparison_summary(self, all_results: Dict) -> Dict:
        """Create comparison summary across scenarios"""
        comparison = {}
        
        for scenario_name, result in all_results.items():
            if 'statistics' in result:
                stats = result['statistics']
                comparison[scenario_name] = {
                    'cost_mean': stats['cost']['mean'],
                    'cost_std': stats['cost']['std'],
                    'cost_cv': stats['cost']['cv'],
                    'grid_import_mean': stats['energy']['grid_import']['mean'],
                    'grid_import_std': stats['energy']['grid_import']['std'],
                    'self_sufficiency_mean': stats['energy']['self_sufficiency']['mean'],
                    'self_sufficiency_std': stats['energy']['self_sufficiency']['std'],
                    'n_successful_runs': stats['n_successful_runs']
                }
        
        return comparison
    
    def _save_comparison_csv(self, comparison: Dict, csv_path: Path):
        """Save comparison as CSV"""
        df = pd.DataFrame.from_dict(comparison, orient='index')
        df.index.name = 'scenario'
        df.to_csv(csv_path)
        print(f"📊 Comparison saved to: {csv_path}")
    
    def _print_summary(self, scenario_name: str, statistics: Dict):
        """Print summary statistics"""
        if not statistics:
            print(f"❌ No statistics available for {scenario_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"STOCHASTIC SUMMARY: {scenario_name}")
        print(f"{'='*60}")
        print(f"Successful runs: {statistics['n_successful_runs']}/{self.n_runs}")
        
        if 'cost' in statistics:
            cost = statistics['cost']
            print(f"\nCost Statistics:")
            print(f"  Mean:  ${cost['mean']:.2f}")
            print(f"  Std:   ${cost['std']:.2f}")
            print(f"  Range: ${cost['min']:.2f} - ${cost['max']:.2f}")
            print(f"  CV:    {cost['cv']:.1f}%")
        
        if 'energy' in statistics:
            grid_import = statistics['energy']['grid_import']
            self_suff = statistics['energy']['self_sufficiency']
            print(f"\nGrid Import Statistics:")
            print(f"  Mean:  {grid_import['mean']:.1f} kWh")
            print(f"  Std:   {grid_import['std']:.1f} kWh")
            print(f"  CV:    {grid_import['cv']:.1f}%")
            
            print(f"\nSelf-Sufficiency Statistics:")
            print(f"  Mean:  {self_suff['mean']:.1f}%")
            print(f"  Std:   {self_suff['std']:.1f}%")
        
        print(f"{'='*60}")
    
    def _print_comparison(self, comparison: Dict):
        """Print comparison across scenarios"""
        print("\nCost Performance (Mean ± Std):")
        for scenario, stats in comparison.items():
            print(f"  {scenario:25s}: ${stats['cost_mean']:.2f} ± ${stats['cost_std']:.2f} (CV: {stats['cost_cv']:.1f}%)")
        
        print("\nGrid Import (Mean ± Std):")
        for scenario, stats in comparison.items():
            print(f"  {scenario:25s}: {stats['grid_import_mean']:.1f} ± {stats['grid_import_std']:.1f} kWh")


def main():
    """Main execution"""
    print("\n🎲 WAIPIRO MICROGRID - STOCHASTIC ANALYSIS")
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    
    # Stochastic parameters
    N_RUNS = 10  # Configurable number of runs
    LOAD_NOISE_STD = 0.10  # 10% standard deviation for loads
    PV_NOISE_STD = 0.20    # 20% standard deviation for PV
    SEED = 42
    
    # Initialize runner
    runner = StochasticSimulationRunner(
        base_dir=base_dir,
        n_runs=N_RUNS,
        load_noise_std=LOAD_NOISE_STD,
        pv_noise_std=PV_NOISE_STD,
        seed=SEED
    )
    
    # Define scenarios to analyze
    scenarios_to_analyze = [
        'local_summer_no_marae',
        'tou_summer_no_marae', 
        'centralized_summer_no_marae'
    ]
    
    print(f"\nRunning stochastic analysis for {len(scenarios_to_analyze)} scenarios...")
    print(f"Parameters: {N_RUNS} runs, {LOAD_NOISE_STD:.1%} load noise, {PV_NOISE_STD:.1%} PV noise")
    
    # Run analysis
    all_results = runner.run_multiple_scenarios(scenarios_to_analyze, n_steps=24)
    
    print(f"\n✅ STOCHASTIC ANALYSIS COMPLETE!")
    print(f"Results saved to: {runner.results_dir}")
    print(f"\nNext steps:")
    print(f"  - Review {runner.results_dir}/stochastic_comparison.csv")
    print(f"  - Analyze individual scenario results in {runner.results_dir}/")


if __name__ == "__main__":
    main()