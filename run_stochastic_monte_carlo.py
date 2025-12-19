#!/usr/bin/env python3
"""
Run Stochastic Monte Carlo Analysis

Tests controllers on multiple stochastic scenarios with Gaussian noise.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.network_manager import NetworkManager
from scenarios.scenario_manager import ScenarioManager
from simulation.simulator import GridSimulator
from utils.cost_calculator import CostCalculator
from utils.stochastic_profiles import generate_stochastic_scenario_batch


def run_stochastic_analysis(
    base_scenario: str = "local_summer_no_marae",
    n_scenarios: int = 10,
    output_dir: str = "results/stochastic"
):
    """
    Run stochastic analysis on a base scenario
    
    Parameters:
    -----------
    base_scenario : str
        Base scenario ID (e.g., 'local_winter_no_marae')
    n_scenarios : int
        Number of stochastic scenarios to generate (default: 10)
    output_dir : str
        Where to save results
    """
    
    print(f"\n{'='*70}")
    print(f"STOCHASTIC MONTE CARLO ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Base scenario: {base_scenario}")
    print(f"Number of scenarios: {n_scenarios}")
    print(f"Load noise: ±10% (1 std dev)")
    print(f"PV noise: ±20% (1 std dev)")
    
    # Initialize
    data_dir = Path("venv/data")
    config_path = Path("config/scenario_definitions.yaml")
    
    network_mgr = NetworkManager(data_dir)
    network_mgr.build_base_network()
    
    scenario_mgr = ScenarioManager(config_path, network_mgr, data_dir)
    simulator = GridSimulator(network_mgr, scenario_mgr, data_dir)
    cost_calculator = CostCalculator(config_path=config_path)
    
    # Determine season and marae
    if 'winter' in base_scenario:
        season = 'winter'
        pv_column = 'capacity_factor_winter'
    else:
        season = 'summer'
        pv_column = 'capacity_factor_summer'
    
    # More specific logic to distinguish no_marae vs marae
    if '_no_marae' in base_scenario:
        sheet_name = f"{season}_no_marae"
    elif '_marae' in base_scenario:
        sheet_name = f"{season}_marae"
    else:
        sheet_name = f"{season}_no_marae"  # Default
    
    # Load base profiles
    print(f"\n📂 Loading base profiles...")
    base_load_df = pd.read_excel(
        data_dir / "loads_profile.xlsx",
        sheet_name=sheet_name,
        index_col=0
    )
    
    base_pv_df = pd.read_csv(
        data_dir / "pv_standard_profile.csv",
        delimiter=';',
        index_col=0
    )
    
    print(f"   Load profile: {sheet_name}")
    print(f"   PV profile: {pv_column}")
    
    # Generate stochastic scenarios
    print(f"\n🎲 Generating {n_scenarios} stochastic scenarios...")
    stochastic_scenarios = generate_stochastic_scenario_batch(
        base_load_df=base_load_df,
        base_pv_df=base_pv_df,
        pv_column=pv_column,
        n_scenarios=n_scenarios,
        load_noise_std=0.10,  # ±10%
        pv_noise_std=0.20,    # ±20%
        seed=42
    )
    
    # Show statistics
    base_load_total = base_load_df.sum().sum()
    stochastic_load_totals = [s['load_df'].sum().sum() for s in stochastic_scenarios]
    
    print(f"\n📊 Stochastic Profile Statistics:")
    print(f"   Base load total:      {base_load_total:.1f} kWh")
    print(f"   Stochastic mean:      {np.mean(stochastic_load_totals):.1f} kWh")
    print(f"   Stochastic std dev:   {np.std(stochastic_load_totals):.1f} kWh")
    print(f"   Range:                [{np.min(stochastic_load_totals):.1f}, "
          f"{np.max(stochastic_load_totals):.1f}] kWh")
    
    # Run simulations
    results = []
    
    for i, scenario in enumerate(stochastic_scenarios):
        print(f"\n{'='*70}")
        print(f"Running Scenario {i+1}/{n_scenarios}")
        print(f"{'='*70}")
        
        # Reset network
        network_mgr.reset_network()
        
        # Apply scenario with custom profiles
        scenario_mgr.apply_scenario_with_custom_profiles(
            scenario_name=base_scenario,
            load_profiles=scenario['load_df'],
            pv_profiles=scenario['pv_df']
        )
        
        # Run simulation
        try:
            # Note: We skip calling simulator.run_scenario() because it would 
            # re-apply the original scenario and overwrite our custom profiles
            
            # Instead, directly run the time series simulation
            import pandapower as pp
            from pandapower.timeseries import run_timeseries
            from pandapower.timeseries.output_writer import OutputWriter
            
            net = scenario_mgr.network_manager.current_net
            results_path = Path(f"{output_dir}/scenario_{i:03d}") / base_scenario
            results_path.mkdir(parents=True, exist_ok=True)
            
            print(f"🔍 DEBUG: Results will be saved to: {results_path}")
            print(f"⏱️  Timesteps: 24")
            print(f"🎛️  Active controllers: {len(net.controller)}")
            
            # Configure output writer
            ow = OutputWriter(net, time_steps=range(24),
                             output_path=results_path, 
                             output_file_type='.csv')
            
            # Log relevant variables
            ow.log_variable('res_load', 'p_mw')
            ow.log_variable('res_sgen', 'p_mw') 
            ow.log_variable('res_storage', 'p_mw')
            ow.log_variable('res_storage', 'soc_percent')
            ow.log_variable('res_bus', 'vm_pu')
            ow.log_variable('res_ext_grid', 'p_mw')
            ow.log_variable('res_trafo', 'loading_percent')
            ow.log_variable('res_line', 'loading_percent')
            
            # Run time series simulation
            run_timeseries(net, time_steps=range(24),
                          run_function=pp.runpp,
                          continue_on_divergence=True,
                          verbose=False)
            
            # Create result dictionary like the simulator does
            result = {
                'data': {},  # We'll populate this if needed
                'network_state': net,
                'output_path': results_path
            }
            
            print("✅ Simulation completed successfully!")
            
            # Calculate costs
            cost_df, cost_summary = cost_calculator.analyze_scenario(
                result['output_path']
            )
            
            # Calculate voltage stability metrics
            voltage_data = pd.read_csv(results_path / 'res_bus' / 'vm_pu.csv', sep=';', index_col=0)
            min_voltage = voltage_data.min().min()
            max_voltage = voltage_data.max().max()
            avg_voltage_dev = (voltage_data - 1.0).abs().mean().mean()
            max_voltage_dev = (voltage_data - 1.0).abs().max().max()
            
            # Calculate transformer loading
            trafo_loading = pd.read_csv(results_path / 'res_trafo' / 'loading_percent.csv', sep=';', index_col=0)
            max_trafo_loading = trafo_loading.max().max()
            avg_trafo_loading = trafo_loading.mean().mean()
            
            # Store results
            results.append({
                'scenario_id': i,
                'total_cost': cost_summary['total_cost'],
                'total_import_kwh': cost_summary['total_import_kwh'],
                'total_export_kwh': cost_summary['total_export_kwh'],
                'peak_import_cost': cost_summary['peak_import_cost'],
                'offpeak_import_cost': cost_summary['offpeak_import_cost'],
                'export_revenue': cost_summary['total_export_revenue'],
                'min_voltage_pu': min_voltage,
                'max_voltage_pu': max_voltage,
                'avg_voltage_deviation': avg_voltage_dev,
                'max_voltage_deviation': max_voltage_dev,
                'max_trafo_loading_pct': max_trafo_loading,
                'avg_trafo_loading_pct': avg_trafo_loading
            })
            
            print(f"✅ Scenario {i+1}: Cost = ${cost_summary['total_cost']:.2f}")
            
        except Exception as e:
            print(f"❌ Scenario {i+1} failed: {e}")
            results.append({
                'scenario_id': i,
                'total_cost': np.nan,
                'total_import_kwh': np.nan,
                'total_export_kwh': np.nan,
                'peak_import_cost': np.nan,
                'offpeak_import_cost': np.nan,
                'export_revenue': np.nan,
                'min_voltage_pu': np.nan,
                'max_voltage_pu': np.nan,
                'avg_voltage_deviation': np.nan,
                'max_voltage_deviation': np.nan,
                'max_trafo_loading_pct': np.nan,
                'avg_trafo_loading_pct': np.nan
            })
    
    # Analyze results
    print(f"\n{'='*70}")
    print("STOCHASTIC ANALYSIS SUMMARY")
    print(f"{'='*70}\n")
    
    df_results = pd.DataFrame(results)
    df_valid = df_results.dropna(subset=['total_cost'])
    
    # Cost statistics
    costs = df_valid['total_cost'].values
    imports = df_valid['total_import_kwh'].values
    exports = df_valid['total_export_kwh'].values
    
    print("Cost Statistics:")
    print(f"  Mean:      ${np.mean(costs):.2f}")
    print(f"  Std Dev:   ${np.std(costs):.2f}")
    print(f"  Min:       ${np.min(costs):.2f}")
    print(f"  Max:       ${np.max(costs):.2f}")
    print(f"  95% CI:    ${np.percentile(costs, 2.5):.2f} - ${np.percentile(costs, 97.5):.2f}")
    print()
    
    print("Grid Import/Export Statistics:")
    print(f"  Mean Import:      {np.mean(imports):.1f} kWh")
    print(f"  Std Dev Import:   {np.std(imports):.1f} kWh")
    print(f"  Mean Export:      {np.mean(exports):.1f} kWh")
    print(f"  Std Dev Export:   {np.std(exports):.1f} kWh")
    print()
    
    print("Voltage Stability:")
    print(f"  Min Voltage:      {df_valid['min_voltage_pu'].min():.4f} p.u.")
    print(f"  Max Voltage:      {df_valid['max_voltage_pu'].max():.4f} p.u.")
    print(f"  Avg Deviation:    {df_valid['avg_voltage_deviation'].mean():.4f} p.u.")
    print(f"  Max Deviation:    {df_valid['max_voltage_deviation'].max():.4f} p.u.")
    print()
    
    print("Transformer Loading:")
    print(f"  Max Loading:      {df_valid['max_trafo_loading_pct'].max():.1f}%")
    print(f"  Avg Loading:      {df_valid['avg_trafo_loading_pct'].mean():.1f}%")
    print()
    
    print("Import Statistics:")
    print(f"  Mean:      {np.mean(imports):.1f} kWh")
    print(f"  Std Dev:   {np.std(imports):.1f} kWh")
    print()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / f"{base_scenario}_stochastic_summary.csv"
    df_results.to_csv(summary_file, index=False)
    
    print(f"💾 Results saved to: {summary_file}")
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stochastic Monte Carlo analysis')
    parser.add_argument('--scenario', type=str, default='local_winter_no_marae',
                       help='Base scenario to test')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of stochastic scenarios')
    parser.add_argument('--output', type=str, default='results/stochastic',
                       help='Output directory')
    
    args = parser.parse_args()
    
    results = run_stochastic_analysis(
        base_scenario=args.scenario,
        n_scenarios=args.n,
        output_dir=args.output
    )