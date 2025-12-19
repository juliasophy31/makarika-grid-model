# run_sensitivity_analysis.py
# Sensitivity analysis for TOU battery control parameters

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import yaml

from models.network_manager import NetworkManager
from scenarios.scenario_manager import ScenarioManager
from simulation.simulator import GridSimulator
from utils.cost_calculator import CostCalculator


def run_sensitivity_analysis(
    base_scenario='tou_summer_no_marae',
    offpeak_threshold_range=None,
    charge_rate_range=None,
    n_steps=24,
    output_dir='results/sensitivity'
):
    """
    Run sensitivity analysis for TOU battery parameters

    Args:
        base_scenario: Base scenario to modify (should be a TOU scenario)
        offpeak_threshold_range: List of offpeak_discharge_threshold values to test
        charge_rate_range: List of offpeak_import_charge_rate values to test
        n_steps: Number of timesteps
        output_dir: Directory to save results

    Returns:
        DataFrame with all results
    """

    # Default parameter ranges if not specified
    if offpeak_threshold_range is None:
        offpeak_threshold_range = [50, 60, 70, 80, 90, 100]

    if charge_rate_range is None:
        charge_rate_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS FOR TOU BATTERY PARAMETERS")
    print("="*80)
    print(f"\nBase Scenario: {base_scenario}")
    print(f"Off-peak Discharge Threshold Range: {offpeak_threshold_range}")
    print(f"Import Charge Rate Range: {charge_rate_range}")
    print(f"Total combinations: {len(offpeak_threshold_range) * len(charge_rate_range)}")
    print("="*80 + "\n")

    # Setup paths
    data_dir = Path("venv/data/")
    config_path = Path("config/scenario_definitions.yaml")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("🔧 Initializing simulation framework...\n")
    network_mgr = NetworkManager(data_dir)
    network_mgr.build_base_network()

    scenario_mgr = ScenarioManager(config_path, network_mgr, data_dir)
    simulator = GridSimulator(network_mgr, scenario_mgr, data_dir)
    cost_calculator = CostCalculator(config_path=config_path)

    # Storage for all results
    all_results = []

    # Counter for progress tracking
    total_runs = len(offpeak_threshold_range) * len(charge_rate_range)
    current_run = 0

    # Run sensitivity analysis
    for threshold in offpeak_threshold_range:
        for charge_rate in charge_rate_range:
            current_run += 1

            print("\n" + "="*80)
            print(f"RUN {current_run}/{total_runs}")
            print(f"  offpeak_discharge_threshold: {threshold}%")
            print(f"  offpeak_import_charge_rate: {charge_rate}")
            print("="*80)

            start_time = time.time()

            try:
                # Reset network for clean state
                network_mgr.reset_network()

                # Load fresh config for each run
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Modify the scenario's market configuration
                if 'market' not in config['scenarios'][base_scenario]:
                    config['scenarios'][base_scenario]['market'] = {}

                # Update the parameters
                config['scenarios'][base_scenario]['market']['offpeak_discharge_threshold'] = threshold
                config['scenarios'][base_scenario]['market']['offpeak_import_charge_rate'] = charge_rate

                # Also update in defaults if it exists
                if 'market' in config.get('defaults', {}):
                    config['defaults']['market']['offpeak_discharge_threshold'] = threshold
                    config['defaults']['market']['offpeak_import_charge_rate'] = charge_rate

                # Save modified config to temporary file
                temp_config_path = Path("config/scenario_definitions_temp.yaml")
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                # Create NEW scenario manager with modified config (important - fresh instance)
                scenario_mgr = ScenarioManager(temp_config_path, network_mgr, data_dir)
                simulator = GridSimulator(network_mgr, scenario_mgr, data_dir)

                # Run simulation
                result = simulator.run_scenario(base_scenario, n_steps=n_steps)

                # Calculate costs
                cost_df, cost_summary = cost_calculator.analyze_scenario(result['output_path'])

                # Extract key metrics
                data = result['data']

                metrics = {
                    'offpeak_discharge_threshold': threshold,
                    'offpeak_import_charge_rate': charge_rate,

                    # Energy metrics
                    'total_load_kwh': data['load'].sum(),
                    'total_pv_kwh': data['pv_generation'].sum(),
                    'battery_discharge_kwh': data['battery_power'][data['battery_power'] > 0].sum(),
                    'battery_charge_kwh': abs(data['battery_power'][data['battery_power'] < 0].sum()),
                    'grid_import_kwh': data['grid_power'][data['grid_power'] > 0].sum(),
                    'grid_export_kwh': abs(data['grid_power'][data['grid_power'] < 0].sum()),

                    # SOC metrics
                    'avg_soc': data['battery_soc'].mean(),
                    'min_soc': data['battery_soc'].min(),
                    'max_soc': data['battery_soc'].max(),
                    'final_soc': data['battery_soc'].iloc[-1],

                    # Cost metrics
                    'total_cost': cost_summary.get('total_cost', 0),
                    'import_cost': cost_summary.get('total_import_cost', 0),
                    'export_revenue': cost_summary.get('total_export_revenue', 0),
                    'net_cost': cost_summary.get('total_cost', 0),

                    # Peak/off-peak breakdown
                    'peak_import_kwh': cost_summary.get('peak_import_kwh', 0),
                    'offpeak_import_kwh': cost_summary.get('offpeak_import_kwh', 0),
                    'peak_import_cost': cost_summary.get('peak_import_cost', 0),
                    'offpeak_import_cost': cost_summary.get('offpeak_import_cost', 0),

                    # Efficiency metrics
                    'self_sufficiency_pct': (data['pv_generation'].sum() / data['load'].sum() * 100) if data['load'].sum() > 0 else 0,
                    'battery_efficiency_pct': (data['battery_power'][data['battery_power'] > 0].sum() /
                                               abs(data['battery_power'][data['battery_power'] < 0].sum()) * 100)
                                               if data['battery_power'][data['battery_power'] < 0].sum() != 0 else 0
                }

                all_results.append(metrics)

                # Clean up temp file
                if temp_config_path.exists():
                    temp_config_path.unlink()

                elapsed = time.time() - start_time
                print(f"✅ Completed in {elapsed:.1f}s")
                print(f"   Total Cost: ${metrics['net_cost']:.2f}")
                print(f"   Grid Import: {metrics['grid_import_kwh']:.1f} kWh")
                print(f"   Avg SOC: {metrics['avg_soc']:.1f}%")

            except Exception as e:
                print(f"❌ Run failed: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETED")
    print("="*80 + "\n")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save results
    csv_path = output_path / f"sensitivity_results_{base_scenario}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"📊 Results saved to: {csv_path}\n")

    return df_results


def create_sensitivity_plots(df_results, base_scenario, output_dir='results/sensitivity'):
    """
    Create visualization plots for sensitivity analysis results

    Args:
        df_results: DataFrame with sensitivity analysis results
        base_scenario: Name of base scenario
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n📊 Creating sensitivity analysis visualizations...\n")

    # ============================================================================
    # 1. HEATMAP: Net Cost vs Parameters
    # ============================================================================
    print("  Creating cost heatmap...")

    # Pivot data for heatmap
    pivot_cost = df_results.pivot(
        index='offpeak_discharge_threshold',
        columns='offpeak_import_charge_rate',
        values='net_cost'
    )

    fig_heatmap_cost = go.Figure(data=go.Heatmap(
        z=pivot_cost.values,
        x=pivot_cost.columns,
        y=pivot_cost.index,
        colorscale='RdYlGn_r',  # Red = high cost, Green = low cost
        text=np.round(pivot_cost.values, 2),
        texttemplate='$%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Net Cost ($)")
    ))

    fig_heatmap_cost.update_layout(
        title=f"Sensitivity Analysis: Net Cost<br>{base_scenario}",
        xaxis_title="Off-Peak Import Charge Rate (C-factor)",
        yaxis_title="Off-Peak Discharge Threshold (%)",
        height=600
    )

    fig_heatmap_cost.write_html(output_path / f"heatmap_cost_{base_scenario}.html")
    print(f"    ✓ Saved: {output_path / f'heatmap_cost_{base_scenario}.html'}")

    # ============================================================================
    # 2. HEATMAP: Grid Import vs Parameters
    # ============================================================================
    print("  Creating grid import heatmap...")

    pivot_import = df_results.pivot(
        index='offpeak_discharge_threshold',
        columns='offpeak_import_charge_rate',
        values='grid_import_kwh'
    )

    fig_heatmap_import = go.Figure(data=go.Heatmap(
        z=pivot_import.values,
        x=pivot_import.columns,
        y=pivot_import.index,
        colorscale='Blues',
        text=np.round(pivot_import.values, 1),
        texttemplate='%{text} kWh',
        textfont={"size": 10},
        colorbar=dict(title="Grid Import (kWh)")
    ))

    fig_heatmap_import.update_layout(
        title=f"Sensitivity Analysis: Grid Import<br>{base_scenario}",
        xaxis_title="Off-Peak Import Charge Rate (C-factor)",
        yaxis_title="Off-Peak Discharge Threshold (%)",
        height=600
    )

    fig_heatmap_import.write_html(output_path / f"heatmap_import_{base_scenario}.html")
    print(f"    ✓ Saved: {output_path / f'heatmap_import_{base_scenario}.html'}")

    # ============================================================================
    # 3. HEATMAP: Average SOC vs Parameters
    # ============================================================================
    print("  Creating average SOC heatmap...")

    pivot_soc = df_results.pivot(
        index='offpeak_discharge_threshold',
        columns='offpeak_import_charge_rate',
        values='avg_soc'
    )

    fig_heatmap_soc = go.Figure(data=go.Heatmap(
        z=pivot_soc.values,
        x=pivot_soc.columns,
        y=pivot_soc.index,
        colorscale='Greens',
        text=np.round(pivot_soc.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Average SOC (%)")
    ))

    fig_heatmap_soc.update_layout(
        title=f"Sensitivity Analysis: Average SOC<br>{base_scenario}",
        xaxis_title="Off-Peak Import Charge Rate (C-factor)",
        yaxis_title="Off-Peak Discharge Threshold (%)",
        height=600
    )

    fig_heatmap_soc.write_html(output_path / f"heatmap_soc_{base_scenario}.html")
    print(f"    ✓ Saved: {output_path / f'heatmap_soc_{base_scenario}.html'}")

    # ============================================================================
    # 3b. HEATMAP: Final SOC vs Parameters
    # ============================================================================
    print("  Creating final SOC heatmap...")

    pivot_final_soc = df_results.pivot(
        index='offpeak_discharge_threshold',
        columns='offpeak_import_charge_rate',
        values='final_soc'
    )

    fig_heatmap_final_soc = go.Figure(data=go.Heatmap(
        z=pivot_final_soc.values,
        x=pivot_final_soc.columns,
        y=pivot_final_soc.index,
        colorscale='RdYlGn',
        text=np.round(pivot_final_soc.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Final SOC (%)")
    ))

    fig_heatmap_final_soc.update_layout(
        title=f"Sensitivity Analysis: Final SOC (Hour 23)<br>{base_scenario}",
        xaxis_title="Off-Peak Import Charge Rate (C-factor)",
        yaxis_title="Off-Peak Discharge Threshold (%)",
        height=600
    )

    fig_heatmap_final_soc.write_html(output_path / f"heatmap_final_soc_{base_scenario}.html")
    print(f"    ✓ Saved: {output_path / f'heatmap_final_soc_{base_scenario}.html'}")

    # ============================================================================
    # 3c. CSV: Final SOC Table
    # ============================================================================
    print("  Creating final SOC table...")
    
    final_soc_table = df_results.pivot(
        index='offpeak_discharge_threshold',
        columns='offpeak_import_charge_rate',
        values='final_soc'
    )
    
    final_soc_path = output_path / f"final_soc_table_{base_scenario}.csv"
    final_soc_table.to_csv(final_soc_path)
    print(f"    ✓ Saved: {final_soc_path}")

    # ============================================================================
    # 4. LINE PLOTS: Parameter Sensitivity Curves
    # ============================================================================
    print("  Creating sensitivity curves...")

    fig_curves = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Net Cost vs Off-Peak Discharge Threshold",
            "Net Cost vs Import Charge Rate",
            "Grid Import vs Off-Peak Discharge Threshold",
            "Grid Import vs Import Charge Rate"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # Group by charge rate for threshold sensitivity
    charge_rates = sorted(df_results['offpeak_import_charge_rate'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, rate in enumerate(charge_rates):
        subset = df_results[df_results['offpeak_import_charge_rate'] == rate]
        subset = subset.sort_values('offpeak_discharge_threshold')

        # Cost vs threshold
        fig_curves.add_trace(go.Scatter(
            x=subset['offpeak_discharge_threshold'],
            y=subset['net_cost'],
            name=f"C-rate={rate}",
            line=dict(color=colors[idx % len(colors)], width=2),
            mode='lines+markers'
        ), row=1, col=1)

        # Import vs threshold
        fig_curves.add_trace(go.Scatter(
            x=subset['offpeak_discharge_threshold'],
            y=subset['grid_import_kwh'],
            name=f"C-rate={rate}",
            line=dict(color=colors[idx % len(colors)], width=2),
            mode='lines+markers',
            showlegend=False
        ), row=2, col=1)

    # Group by threshold for charge rate sensitivity
    thresholds = sorted(df_results['offpeak_discharge_threshold'].unique())

    for idx, threshold in enumerate(thresholds):
        subset = df_results[df_results['offpeak_discharge_threshold'] == threshold]
        subset = subset.sort_values('offpeak_import_charge_rate')

        # Cost vs charge rate
        fig_curves.add_trace(go.Scatter(
            x=subset['offpeak_import_charge_rate'],
            y=subset['net_cost'],
            name=f"Threshold={threshold}%",
            line=dict(width=2),
            mode='lines+markers',
            showlegend=False
        ), row=1, col=2)

        # Import vs charge rate
        fig_curves.add_trace(go.Scatter(
            x=subset['offpeak_import_charge_rate'],
            y=subset['grid_import_kwh'],
            name=f"Threshold={threshold}%",
            line=dict(width=2),
            mode='lines+markers',
            showlegend=False
        ), row=2, col=2)

    fig_curves.update_xaxes(title_text="Off-Peak Discharge Threshold (%)", row=1, col=1)
    fig_curves.update_xaxes(title_text="Import Charge Rate", row=1, col=2)
    fig_curves.update_xaxes(title_text="Off-Peak Discharge Threshold (%)", row=2, col=1)
    fig_curves.update_xaxes(title_text="Import Charge Rate", row=2, col=2)

    fig_curves.update_yaxes(title_text="Net Cost ($)", row=1, col=1)
    fig_curves.update_yaxes(title_text="Net Cost ($)", row=1, col=2)
    fig_curves.update_yaxes(title_text="Grid Import (kWh)", row=2, col=1)
    fig_curves.update_yaxes(title_text="Grid Import (kWh)", row=2, col=2)

    fig_curves.update_layout(
        title_text=f"Sensitivity Curves: {base_scenario}",
        height=800,
        hovermode='x unified'
    )

    fig_curves.write_html(output_path / f"sensitivity_curves_{base_scenario}.html")
    print(f"    ✓ Saved: {output_path / f'sensitivity_curves_{base_scenario}.html'}")

    # ============================================================================
    # 5. SUMMARY TABLE: Best Parameter Combinations
    # ============================================================================
    print("  Creating summary table...")

    # Find optimal combinations
    best_cost = df_results.loc[df_results['net_cost'].idxmin()]
    best_import = df_results.loc[df_results['grid_import_kwh'].idxmin()]
    best_soc = df_results.loc[df_results['avg_soc'].idxmax()]

    summary_data = {
        'Objective': ['Minimize Cost', 'Minimize Grid Import', 'Maximize Avg SOC'],
        'Threshold (%)': [best_cost['offpeak_discharge_threshold'],
                          best_import['offpeak_discharge_threshold'],
                          best_soc['offpeak_discharge_threshold']],
        'Charge Rate': [best_cost['offpeak_import_charge_rate'],
                        best_import['offpeak_import_charge_rate'],
                        best_soc['offpeak_import_charge_rate']],
        'Net Cost ($)': [best_cost['net_cost'], best_import['net_cost'], best_soc['net_cost']],
        'Grid Import (kWh)': [best_cost['grid_import_kwh'], best_import['grid_import_kwh'], best_soc['grid_import_kwh']],
        'Avg SOC (%)': [best_cost['avg_soc'], best_import['avg_soc'], best_soc['avg_soc']]
    }

    df_summary = pd.DataFrame(summary_data)

    summary_path = output_path / f"optimal_parameters_{base_scenario}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"    ✓ Saved: {summary_path}")

    print("\n" + "="*80)
    print("OPTIMAL PARAMETER COMBINATIONS")
    print("="*80 + "\n")
    print(df_summary.to_string(index=False))
    print("\n" + "="*80 + "\n")

    print(f"\n✅ All visualizations saved to: {output_path.absolute()}\n")


def main():
    """Main execution"""
    print("\n🔌 WAIPIRO MICROGRID - TOU SENSITIVITY ANALYSIS\n")

    # Configuration
    base_scenario = 'tou_summer_no_marae'
    offpeak_threshold_range = [50, 60, 70, 80, 90, 100]
    charge_rate_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Run sensitivity analysis
    df_results = run_sensitivity_analysis(
        base_scenario=base_scenario,
        offpeak_threshold_range=offpeak_threshold_range,
        charge_rate_range=charge_rate_range,
        n_steps=24
    )

    # Create visualizations
    create_sensitivity_plots(df_results, base_scenario)

    print("\n✅ SENSITIVITY ANALYSIS COMPLETE!")
    print("\n📁 Results saved to:")
    print("   • results/sensitivity/")
    print("\n📊 Generated files:")
    print("   • sensitivity_results_*.csv - Raw data")
    print("   • heatmap_cost_*.html - Cost heatmap")
    print("   • heatmap_import_*.html - Grid import heatmap")
    print("   • heatmap_soc_*.html - Average SOC heatmap")
    print("   • heatmap_final_soc_*.html - Final SOC heatmap")
    print("   • final_soc_table_*.csv - Final SOC table")
    print("   • sensitivity_curves_*.html - Sensitivity curves")
    print("   • optimal_parameters_*.csv - Optimal parameter combinations\n")


if __name__ == "__main__":
    main()
