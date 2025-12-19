# utils/cost_calculator.py - Cost calculation for microgrid scenarios
"""
Cost Calculator - Economic Analysis for Microgrid Simulations
==============================================================

This module calculates electricity costs and revenue for microgrid scenarios
based on grid import/export patterns and time-of-use (TOU) tariff structures.

What it does:
-------------
- Calculates electricity costs with time-of-use pricing (peak/off-peak)
- Computes export revenue from excess generation
- Provides timestep-by-timestep cost breakdown
- Generates daily/scenario summary metrics
- Compares costs across multiple scenarios
- Exports results to CSV files


How to use:
-----------

OPTION 1: Analyze a single scenario
    from utils.cost_calculator import CostCalculator
    from pathlib import Path
    
    calculator = CostCalculator(config_path=Path("config/scenario_definitions.yaml"))
    cost_df, summary = calculator.analyze_scenario(Path("results/my_scenario"))
    
    print(f"Total cost: ${summary['total_cost']:.2f}")
    print(f"Peak import cost: ${summary['peak_import_cost']:.2f}")

OPTION 2: Compare multiple scenarios
    scenarios = {
        "Local Control": Path("results/local_summer"),
        "TOU Control": Path("results/tou_summer"),
        "Optimized": Path("results/optimized_summer")
    }
    
    comparison = calculator.compare_scenarios(scenarios)
    comparison.to_csv("results/cost_comparison.csv")


Input Requirements:
-------------------
The calculator expects simulation results in pandapower format:
- Grid power data: results/{scenario}/res_ext_grid/p_mw.csv
- Format: CSV with semicolon delimiter
- Columns: timestep x grid connection
- Units: MW (converted internally to kW)
- Sign convention: Positive = import, Negative = export


Integration:
------------
This calculator is used by:
- streamlit_app.py: Display costs in web dashboard

It reads results from:
- GridSimulator output (res_ext_grid/p_mw.csv)
- scenario_definitions.yaml (for default tariffs)


Dependencies:
-------------
- pandas: Data manipulation and CSV I/O
- yaml: Configuration file parsing
- pathlib: File path handling
- typing: Type hints

Notes:
------
- All costs in NZD ($)
- Timestep assumed to be 1 hour (can be adjusted)
- Self-sufficiency calculated as: 1 - (import / total_demand)
- Export revenue reduces total cost (net cost = import_cost - export_revenue)
- Negative total_cost means net profit (rare, usually requires high export)

"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, List


class CostCalculator:
    
    def __init__(
        self,
        config_path: Path = None,
        peak_hours: List[int] = None,
        peak_import_price: float = None,
        offpeak_import_price: float = None,
        export_price: float = None
    ):
        # Load from config if provided
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                defaults = config.get('defaults', {}).get('market', {})

                self.peak_hours = peak_hours or defaults.get('peak_hours', [7, 8, 9, 10, 17, 18, 19, 20])
                self.peak_import_price = peak_import_price or defaults.get('peak_import_price', 0.33)
                self.offpeak_import_price = offpeak_import_price or defaults.get('offpeak_import_price', 0.23)
                self.export_price = export_price or defaults.get('export_price', 0.13)
        else:
            # Use provided values or defaults
            self.peak_hours = peak_hours or [7, 8, 9, 10, 17, 18, 19, 20]
            self.peak_import_price = peak_import_price or 0.33
            self.offpeak_import_price = offpeak_import_price or 0.23
            self.export_price = export_price or 0.13

    def is_peak_hour(self, hour: int) -> bool:
        return hour in self.peak_hours

    def calculate_timestep_cost(
        self,
        hour: int,
        grid_power_kw: float
    ) -> Dict[str, float]:
        # Determine if peak or off-peak
        is_peak = self.is_peak_hour(hour)
        tariff = self.peak_import_price if is_peak else self.offpeak_import_price

        # Separate import and export (assume 1-hour timestep)
        if grid_power_kw > 0:
            # Importing from grid
            import_kwh = grid_power_kw
            export_kwh = 0.0
            import_cost = import_kwh * tariff
            export_revenue = 0.0
        else:
            # Exporting to grid
            import_kwh = 0.0
            export_kwh = abs(grid_power_kw)
            import_cost = 0.0
            export_revenue = export_kwh * self.export_price

        net_cost = import_cost - export_revenue

        return {
            'hour': hour,
            'is_peak': is_peak,
            'import_kwh': import_kwh,
            'export_kwh': export_kwh,
            'import_cost': import_cost,
            'export_revenue': export_revenue,
            'net_cost': net_cost,
            'tariff_used': tariff
        }

    def calculate_scenario_costs(
        self,
        grid_power_series: pd.Series,
        time_hours: range
    ) -> pd.DataFrame:
        results = []

        for i, hour in enumerate(time_hours):
            grid_power_kw = grid_power_series.iloc[i]

            # Calculate hour of day (wrap around for multi-day simulations)
            hour_of_day = hour % 24

            timestep_result = self.calculate_timestep_cost(hour_of_day, grid_power_kw)
            timestep_result['timestep'] = hour

            results.append(timestep_result)

        return pd.DataFrame(results)

    def calculate_daily_summary(
        self,
        cost_df: pd.DataFrame
    ) -> Dict[str, float]:
        # Total import and export
        total_import_kwh = cost_df['import_kwh'].sum()
        total_export_kwh = cost_df['export_kwh'].sum()

        # Total costs and revenue
        total_import_cost = cost_df['import_cost'].sum()
        total_export_revenue = cost_df['export_revenue'].sum()

        # Peak vs off-peak breakdown
        peak_df = cost_df[cost_df['is_peak'] == True]
        offpeak_df = cost_df[cost_df['is_peak'] == False]

        peak_import_cost = peak_df['import_cost'].sum()
        offpeak_import_cost = offpeak_df['import_cost'].sum()

        peak_import_kwh = peak_df['import_kwh'].sum()
        offpeak_import_kwh = offpeak_df['import_kwh'].sum()

        # Total cost (import - export revenue)
        total_cost = total_import_cost - total_export_revenue

        return {
            'total_import_kwh': total_import_kwh,
            'total_export_kwh': total_export_kwh,
            'net_exchange_kwh': total_import_kwh - total_export_kwh,
            'total_import_cost': total_import_cost,
            'total_export_revenue': total_export_revenue,
            'peak_import_kwh': peak_import_kwh,
            'offpeak_import_kwh': offpeak_import_kwh,
            'peak_import_cost': peak_import_cost,
            'offpeak_import_cost': offpeak_import_cost,
            'total_cost': total_cost,
            'self_sufficiency_percent': max(0, 100 * (1 - total_import_kwh / (total_import_kwh + total_export_kwh))) if (total_import_kwh + total_export_kwh) > 0 else 0
        }

    def analyze_scenario(
        self,
        results_path: Path
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        # Load grid power data
        grid_file = results_path / "res_ext_grid" / "p_mw.csv"

        if not grid_file.exists():
            raise FileNotFoundError(f"Grid data not found at {grid_file}")

        grid_data = pd.read_csv(grid_file, index_col=0, delimiter=';')

        # Convert to kW and get first column (main grid connection)
        grid_power_kw = grid_data.iloc[:, 0] * 1000

        # Create time range
        time_hours = range(len(grid_power_kw))

        # Calculate costs
        cost_df = self.calculate_scenario_costs(grid_power_kw, time_hours)

        # Calculate summary
        summary = self.calculate_daily_summary(cost_df)

        return cost_df, summary

    def compare_scenarios(
        self,
        scenario_results: Dict[str, Path]
    ) -> pd.DataFrame:
        comparison_data = []

        for scenario_name, results_path in scenario_results.items():
            try:
                _, summary = self.analyze_scenario(results_path)

                comparison_data.append({
                    'Scenario': scenario_name,
                    'Total Cost ($)': summary['total_cost'],
                    'Import (kWh)': summary['total_import_kwh'],
                    'Export (kWh)': summary['total_export_kwh'],
                    'Net Exchange (kWh)': summary['net_exchange_kwh'],
                    'Import Cost ($)': summary['total_import_cost'],
                    'Export Revenue ($)': summary['total_export_revenue'],
                    'Peak Import Cost ($)': summary['peak_import_cost'],
                    'Off-Peak Import Cost ($)': summary['offpeak_import_cost'],
                    'Self Sufficiency (%)': summary['self_sufficiency_percent']
                })
            except Exception as e:
                print(f"Warning: Could not analyze scenario '{scenario_name}': {e}")

        return pd.DataFrame(comparison_data)

    def export_cost_analysis(
        self,
        cost_df: pd.DataFrame,
        summary: Dict[str, float],
        scenario_name: str = "scenario",
        base_output_dir: Path = Path("results/costs")
    ):
        # Create costs directory structure: results/costs/{scenario_name}/
        scenario_cost_dir = base_output_dir / scenario_name
        scenario_cost_dir.mkdir(parents=True, exist_ok=True)

        # Save timestep costs
        timestep_file = scenario_cost_dir / "cost_timesteps.csv"
        cost_df.to_csv(timestep_file, index=False)

        # Save summary
        summary_file = scenario_cost_dir / "cost_summary.csv"
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['Value']
        summary_df.to_csv(summary_file)

        print(f"✅ Cost analysis exported to {scenario_cost_dir}")

        return scenario_cost_dir


if __name__ == "__main__":
    # Example: Analyze a single scenario
    from pathlib import Path

    results_dir = Path("results/base_local_control")
    config_path = Path("config/scenario_definitions.yaml")

    if results_dir.exists():
        print("Analyzing scenario costs...")

        # Create calculator with config defaults
        calculator = CostCalculator(config_path=config_path)

        print(f"\nUsing tariff structure:")
        print(f"  Peak hours: {calculator.peak_hours}")
        print(f"  Peak import price: ${calculator.peak_import_price}/kWh")
        print(f"  Off-peak import price: ${calculator.offpeak_import_price}/kWh")
        print(f"  Export price: ${calculator.export_price}/kWh")

        # Analyze scenario
        cost_df, summary = calculator.analyze_scenario(results_dir)

        # Print summary
        print("\n" + "="*60)
        print("COST ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Cost:              ${summary['total_cost']:.2f}")
        print(f"Total Import:            {summary['total_import_kwh']:.1f} kWh")
        print(f"Total Export:            {summary['total_export_kwh']:.1f} kWh")
        print(f"Net Exchange:            {summary['net_exchange_kwh']:.1f} kWh")
        print(f"\nImport Cost:             ${summary['total_import_cost']:.2f}")
        print(f"  - Peak Hours:          ${summary['peak_import_cost']:.2f}")
        print(f"  - Off-Peak Hours:      ${summary['offpeak_import_cost']:.2f}")
        print(f"Export Revenue:          ${summary['total_export_revenue']:.2f}")
        print(f"Self Sufficiency:        {summary['self_sufficiency_percent']:.1f}%")
        print("="*60)

        # Export results
        calculator.export_cost_analysis(
            cost_df,
            summary,
            scenario_name="base_local_control"
        )
    else:
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run a simulation first.")
