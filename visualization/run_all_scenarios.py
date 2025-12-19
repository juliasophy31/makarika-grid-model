# run_all_scenarios.py
# Script to run all 12 scenarios sequentially

from pathlib import Path
import pandas as pd
import time
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network_manager import NetworkManager
from scenarios.scenario_manager import ScenarioManager
from simulation.simulator import GridSimulator
from utils.cost_calculator import CostCalculator


def run_all_scenarios(n_steps=24):
    """
    Run all 12 scenarios (local + TOU + optimized) and collect results

    Returns:
        dict: Results for each scenario
        dict: Cost summaries for each scenario
        ScenarioManager: Scenario manager instance
    """
    print("\n" + "="*80)
    print("RUNNING ALL SCENARIOS")
    print("="*80 + "\n")

    # Setup paths (relative to project root)
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "venv/data/"
    config_path = base_dir / "config/scenario_definitions.yaml"

    # Initialize components
    print("Initializing simulation framework...\n")
    network_mgr = NetworkManager(data_dir)
    network_mgr.build_base_network()

    scenario_mgr = ScenarioManager(config_path, network_mgr, data_dir)
    simulator = GridSimulator(network_mgr, scenario_mgr, data_dir)
    cost_calculator = CostCalculator(config_path=config_path)

    # Get all scenario names
    scenarios = scenario_mgr.list_scenarios()
    scenario_ids = [s['id'] for s in scenarios]

    print(f"Found {len(scenario_ids)} scenarios to run:\n")
    for s in scenarios:
        print(f"  - {s['id']}: {s['name']}")
    print()

    # Storage for all results
    all_results = {}
    all_costs = {}

    # Run each scenario
    for idx, scenario_id in enumerate(scenario_ids, 1):
        print("\n" + "="*80)
        print(f"RUNNING SCENARIO {idx}/{len(scenario_ids)}: {scenario_id}")
        print("="*80)

        start_time = time.time()

        try:
            # Reset network for clean state
            network_mgr.reset_network()

            # Run simulation
            result = simulator.run_scenario(scenario_id, n_steps=n_steps)
            all_results[scenario_id] = result

            # Calculate costs
            try:
                cost_df, cost_summary = cost_calculator.analyze_scenario(result['output_path'])
                cost_calculator.export_cost_analysis(cost_df, cost_summary, scenario_name=scenario_id)
                all_costs[scenario_id] = cost_summary
                print(f"Total cost: ${cost_summary['total_cost']:.2f}")
            except Exception as e:
                print(f"Cost calculation failed: {e}")
                all_costs[scenario_id] = None

            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.1f}s")

        except Exception as e:
            print(f"Scenario failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[scenario_id] = None
            all_costs[scenario_id] = None

    print("\n" + "="*80)
    print("ALL SCENARIOS COMPLETED")
    print("="*80 + "\n")

    # Print summary table
    _print_summary_table(all_results, all_costs)

    return all_results, all_costs, scenario_mgr


def _print_summary_table(all_results, all_costs):
    """Print summary table of all scenarios"""
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80 + "\n")

    # Filter valid scenarios
    valid_scenarios = {k: v for k, v in all_results.items() if v is not None}

    if not valid_scenarios:
        print("No valid scenarios to summarize!")
        return

    # Collect data
    summary_data = []
    for scenario_id, result in valid_scenarios.items():
        data = result['data']

        total_load = data['load'].sum()
        total_pv = data['pv_generation'].sum()
        battery_discharge = data['battery_power'][data['battery_power'] > 0].sum()
        battery_charge = abs(data['battery_power'][data['battery_power'] < 0].sum())
        grid_import = data['grid_power'][data['grid_power'] > 0].sum()
        grid_export = abs(data['grid_power'][data['grid_power'] < 0].sum())

        self_sufficiency = (total_pv / total_load * 100) if total_load > 0 else 0

        cost_info = all_costs.get(scenario_id)
        total_cost = cost_info.get('total_cost', 0) if cost_info else 0

        summary_data.append({
            'Scenario': scenario_id,
            'Load (kWh)': round(total_load, 1),
            'PV (kWh)': round(total_pv, 1),
            'Batt Charge (kWh)': round(battery_charge, 1),
            'Batt Discharge (kWh)': round(battery_discharge, 1),
            'Grid Import (kWh)': round(grid_import, 1),
            'Grid Export (kWh)': round(grid_export, 1),
            'Self-Suff (%)': round(self_sufficiency, 1),
            'Cost ($)': round(total_cost, 2)
        })

    df_summary = pd.DataFrame(summary_data)

    # Print to console
    print(df_summary.to_string(index=False))
    print("\n" + "="*80 + "\n")

    # Save to CSV
    output_dir = Path("results/overview")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "simulation_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}\n")


def main():
    """Main execution"""
    print("\n WAIPIRO MICROGRID - ALL SCENARIOS SIMULATION\n")

    # Run all scenarios
    all_results, all_costs, scenario_mgr = run_all_scenarios(n_steps=24)

    print("\nSIMULATION COMPLETE!")
    print("\nResults saved to:")
    print("  - results/<scenario_name>/ (individual scenario results)")
    print("  - results/costs/ (cost analysis)")
    print("  - results/overview/simulation_summary.csv")
    print("\nNext steps:")
    print("  1. Run: python visualization/create_visualizations.py")
    print("  2. Or use: streamlit run streamlit_app.py\n")


if __name__ == "__main__":
    main()