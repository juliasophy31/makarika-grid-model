
"""
Individual Visualization Generator for Microgrid Analysis

This script creates individual visualizations for the
economic and technical performance of the Makarika microgrid simulation project across all scenarios.

Usage:
    python create_visualizations.py
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "venv" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "overview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOADS_FILE = DATA_DIR / "loads_profile.xlsx"
LOADS_NETWORK_FILE = DATA_DIR / "network_waipiro_loads.csv"

# Scenario mapping - ALL 12 scenarios (local + TOU + optimized )
SCENARIO_MAP = {
    "local_winter_no_marae": "local_winter_no_marae",
    "local_winter_marae": "local_winter_marae",
    "local_summer_no_marae": "local_summer_no_marae",
    "local_summer_marae": "local_summer_marae",
    "tou_winter_no_marae": "tou_winter_no_marae",
    "tou_winter_marae": "tou_winter_marae",
    "tou_summer_no_marae": "tou_summer_no_marae",
    "tou_summer_marae": "tou_summer_marae",
    "optimized_winter_no_marae": "optimized_winter_no_marae",
    "optimized_winter_marae": "optimized_winter_marae",
    "optimized_summer_no_marae": "optimized_summer_no_marae",
    "optimized_summer_marae": "optimized_summer_marae"
}

# Sheet name mapping for load profiles (same for all control strategies)
LOAD_SHEET_MAP = {
    "winter_no_marae": "local_winter_no_marae",
    "winter_marae": "local_winter_marae",
    "summer_no_marae": "local_summer_no_marae",
    "summer_marae": "local_summer_marae"
}

# Sheet names in the Excel file
SHEET_NAMES = ["winter_no_marae", "winter_marae", "summer_no_marae", "summer_marae"]

# Color schemes for individual loads (consistent across all scenarios)
INDIVIDUAL_LOAD_COLORS = {
    'load_bus4_kw': '#1f77b4',   # Blue
    'load_bus10_kw': '#ff7f0e',  # Orange
    'load_bus11_kw': '#2ca02c',  # Green
    'load_bus17_kw': '#d62728',  # Red
    'load_bus18_kw': '#9467bd',  # Purple
    'load_bus20_kw': '#8c564b',  # Brown
    'load_bus21_kw': '#e377c2',  # Pink
}

# Color schemes for aggregated PV vs Load
PV_COLORS = {
    'winter_no_marae': "#FF5100",   # Dark Orange
    'winter_marae': "#FF8D47",      # Tomato
    'summer_no_marae': '#FFD700',   # Gold
    'summer_marae': "#FFDD00"       # Yellow
}

AGG_LOAD_COLORS = {
    'winter_no_marae': '#1E90FF',   # Dodger Blue
    'winter_marae': '#4169E1',      # Royal Blue
    'summer_no_marae': '#32CD32',   # Lime Green
    'summer_marae': '#228B22'       # Forest Green
}

# Color schemes for control strategies (used in SOC comparisons)
CONTROL_STRATEGY_COLORS = {
    'local': ['#32CD32', '#228B22'],           # Greens (Lime, Forest)
    'tou': ['#FF8C00', '#CC5500'],             # Oranges (Dark Orange, Burnt Orange)
    'optimized': ['#9370DB', '#6A5ACD'],       # Purples (Medium Purple, Slate Blue
}

# Peak hours for shading
PEAK_HOURS = [7, 8, 9, 10, 17, 18, 19, 20]

# All scenario IDs for iteration
ALL_SCENARIOS = list(SCENARIO_MAP.keys())

# Results directory for reading simulation outputs
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_peak_hour_shading(fig, peak_hours=PEAK_HOURS, row=None, col=None):
    """
    Add shaded rectangles for peak hours to all graphs

    Args:
        fig: Plotly figure object
        peak_hours: List of peak hour indices
        row: Row number for subplot (None for single plot)
        col: Column number for subplot (None for single plot)
    """
    for hour in peak_hours:
        fig.add_vrect(
            x0=hour, x1=hour+1,
            fillcolor="rgba(255, 200, 200, 0.2)",
            layer="below",
            line_width=0,
            row=row, col=col
        )

# ============================================================================
# VISUALIZATION 1: Load Profiles Over Time (All 7 Loads)
# ============================================================================

def create_load_profiles_visualization(sheet_name="winter_no_marae"):
    """
    Create a visualization showing all 7 load profiles over 24 hours

    Args:
        sheet_name: Which scenario sheet to visualize
    """
    print(f"\nCreating load profiles visualization for: {sheet_name}")

    df = pd.read_excel(LOADS_FILE, sheet_name=sheet_name, index_col=0)

    print(f"  Loaded data shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    fig = go.Figure()

    # Group loads by identical values
    load_groups = {}
    for col in df.columns:
        values_tuple = tuple(df[col].values)
        if values_tuple not in load_groups:
            load_groups[values_tuple] = []
        load_groups[values_tuple].append(col)

    # Assign colors
    color_assignment = {}
    available_colors = list(INDIVIDUAL_LOAD_COLORS.values())
    color_idx = 0

    for values_tuple, load_list in load_groups.items():
        assigned_color = available_colors[color_idx % len(available_colors)]
        for load_col in load_list:
            color_assignment[load_col] = assigned_color
        color_idx += 1

    # Add traces
    for col in df.columns:
        color = color_assignment.get(col, '#1f77b4')
        display_name = col.replace('load_', '').replace('_kw', '').replace('_', ' ').title()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=display_name,
            line=dict(color=color, width=2),
            mode='lines+markers',
            marker=dict(size=4)
        ))

    scenario_title = sheet_name.replace('_', ' ').title()

    fig.update_layout(
        title=f"Load Profiles - {scenario_title}",
        xaxis_title="Hour of Day",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        height=600,
        legend=dict(
            title="Load at Bus",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1
        ),
        template="plotly_white"
    )

    fig.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(gridcolor='lightgray', showgrid=True)

    output_file = OUTPUT_DIR / f"load_profiles_{sheet_name}.png"
    fig.write_image(output_file, width=1200, height=600, scale=2)
    print(f"  ✓ Saved: {output_file}")

    return fig


# ============================================================================
# VISUALIZATION 2: Aggregated PV Generation vs Aggregated Load
# ============================================================================

def create_aggregated_pv_load_visualization(scenario_ids=None):
    """
    Create visualization showing aggregated community load and PV generation
    """
    if scenario_ids is None:
        scenario_ids = list(LOAD_SHEET_MAP.keys())

    print(f"\nCreating aggregated PV vs Load visualization...")

    results_base = Path(__file__).parent.parent / "results"
    results = {}

    for sheet_name in scenario_ids:
        scenario_id = LOAD_SHEET_MAP[sheet_name]
        scenario_dir = results_base / scenario_id

        print(f"  Loading scenario: {scenario_id}")

        try:
            load_file = scenario_dir / "res_load" / "p_mw.csv"
            if not load_file.exists():
                print(f"    ⚠ Load file not found: {load_file}")
                continue

            df_load = pd.read_csv(load_file, delimiter=';', index_col=0)
            total_load = df_load.sum(axis=1) * 1000

            pv_file = scenario_dir / "res_sgen" / "p_mw.csv"
            if not pv_file.exists():
                print(f"    ⚠ PV file not found: {pv_file}")
                continue

            df_pv = pd.read_csv(pv_file, delimiter=';', index_col=0)
            total_pv = df_pv.sum(axis=1) * 1000

            results[sheet_name] = {
                'time_hours': range(len(total_load)),
                'load': total_load.values,
                'pv_generation': total_pv.values
            }

            print(f"    ✓ Loaded {len(total_load)} timesteps")

        except Exception as e:
            print(f"    ⚠ Error loading {scenario_id}: {e}")
            continue

    fig = go.Figure()

    display_names = {
        'winter_no_marae': 'Winter - No Marae',
        'winter_marae': 'Winter - Marae Event',
        'summer_no_marae': 'Summer - No Marae',
        'summer_marae': 'Summer - Marae Event'
    }

    # Add PV generation (no_marae only)
    for sheet_name in scenario_ids:
        if sheet_name in results and 'no_marae' in sheet_name:
            data = results[sheet_name]
            pv_color = PV_COLORS.get(sheet_name, '#FFD700')
            display_name = display_names.get(sheet_name, sheet_name)
            gen_display_name = display_name.replace(' - No Marae', '')

            fig.add_trace(go.Scatter(
                x=list(data['time_hours']),
                y=data['pv_generation'],
                name=f"{gen_display_name} - Generation",
                line=dict(dash='dash', color=pv_color, width=3),
                mode='lines',
                legendgroup='generation'
            ))

    # Add loads
    for sheet_name in scenario_ids:
        if sheet_name in results:
            data = results[sheet_name]
            load_color = AGG_LOAD_COLORS.get(sheet_name, '#1E90FF')
            display_name = display_names.get(sheet_name, sheet_name)

            fig.add_trace(go.Scatter(
                x=list(data['time_hours']),
                y=data['load'],
                name=f"{display_name} - Load",
                line=dict(color=load_color, width=3),
                mode='lines',
                legendgroup='load'
            ))

    fig.update_layout(
        title="Aggregated PV Generation vs Community Load - All Scenarios",
        xaxis_title="Hour of Day",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1,
            tracegroupgap=15,
            font=dict(size=11)
        ),
        template="plotly_white"
    )

    fig.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(gridcolor='lightgray', showgrid=True)

    output_file = OUTPUT_DIR / "aggregated_pv_vs_load_all_scenarios.png"
    fig.write_image(output_file, width=1400, height=600, scale=2)
    print(f"  ✓ Saved: {output_file}")

    return fig


# ============================================================================
# VISUALIZATION 3: SOC Profiles for All 12 Scenarios (Separate Graphs)
# ============================================================================

def create_soc_profiles_all_scenarios():
    """
    Create separate SOC profile visualizations for all 12 scenarios
    """
    print(f"\nCreating SOC profile visualizations for all scenarios...")

    results_base = Path(__file__).parent.parent / "results"

    for scenario_id in SCENARIO_MAP.values():
        scenario_dir = results_base / scenario_id

        print(f"  Creating SOC graph for: {scenario_id}")

        try:
            soc_file = scenario_dir / "res_storage" / "soc_percent.csv"
            if not soc_file.exists():
                print(f"    ⚠ SOC file not found: {soc_file}")
                continue

            df_soc = pd.read_csv(soc_file, delimiter=';', index_col=0)

            fig = go.Figure()

            for battery_col in df_soc.columns:
                display_name = battery_col.replace('Battery_', 'Battery ')

                fig.add_trace(go.Scatter(
                    x=df_soc.index,
                    y=df_soc[battery_col],
                    name=display_name,
                    mode='lines+markers',
                    marker=dict(size=4),
                    line=dict(width=2)
                ))

            scenario_title = scenario_id.replace('_', ' ').title()

            fig.update_layout(
                title=f"Battery State of Charge - {scenario_title}",
                xaxis_title="Hour of Day",
                yaxis_title="SOC (%)",
                hovermode='x unified',
                height=600,
                legend=dict(
                    title="Battery",
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="Black",
                    borderwidth=1
                ),
                template="plotly_white"
            )

            fig.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
            fig.update_yaxes(range=[0, 100], gridcolor='lightgray', showgrid=True)

            add_peak_hour_shading(fig)

            output_file = OUTPUT_DIR / f"soc_profile_{scenario_id}.png"
            fig.write_image(output_file, width=1200, height=600, scale=2)
            print(f"    ✓ Saved: {output_file}")

        except Exception as e:
            print(f"    ⚠ Error creating SOC graph for {scenario_id}: {e}")
            continue


# ============================================================================
# VISUALIZATION 4: SOC Comparison - All Three Strategies (Summer No Marae)
# ============================================================================

def create_soc_comparison_all_strategies_summer():
    """
    Create SOC comparison visualization for summer_no_marae scenario
    comparing local, TOU, optimized control strategies
    """
    print(f"\nCreating SOC comparison: All Strategies (Summer No Marae)...")

    results_base = Path(__file__).parent.parent / "results"

    scenarios_to_compare = {
        'local_summer_no_marae': 'Local Control',
        'tou_summer_no_marae': 'TOU Control',
        'optimized_summer_no_marae': 'Optimized Control'
    }

    soc_data = {}

    for scenario_id, display_name in scenarios_to_compare.items():
        scenario_dir = results_base / scenario_id

        print(f"  Loading SOC for: {scenario_id}")

        try:
            soc_file = scenario_dir / "res_storage" / "soc_percent.csv"
            if not soc_file.exists():
                print(f"    ⚠ SOC file not found: {soc_file}")
                continue

            df_soc = pd.read_csv(soc_file, delimiter=';', index_col=0)
            soc_data[scenario_id] = {
                'df': df_soc,
                'display_name': display_name
            }

            print(f"    ✓ Loaded {len(df_soc)} timesteps")

        except Exception as e:
            print(f"    ⚠ Error loading {scenario_id}: {e}")
            continue

    fig = go.Figure()

    # Add traces for all three strategies
    for scenario_id, data_dict in soc_data.items():
        df_soc = data_dict['df']
        control_name = data_dict['display_name']

        # Determine color scheme based on control type
        if 'local' in scenario_id:
            colors = CONTROL_STRATEGY_COLORS['local']
        elif 'tou' in scenario_id:
            colors = CONTROL_STRATEGY_COLORS['tou']
        else:  # optimized
            colors = CONTROL_STRATEGY_COLORS['optimized']

        # Add trace for each battery
        for idx, battery_col in enumerate(df_soc.columns):
            battery_name = battery_col.replace('Battery_', 'Battery ')
            color = colors[idx % len(colors)]

            fig.add_trace(go.Scatter(
                x=df_soc.index,
                y=df_soc[battery_col],
                name=f"{control_name} - {battery_name}",
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(width=2, color=color),
                legendgroup=scenario_id
            ))

    fig.update_layout(
        title="Battery SOC Comparison: All Control Strategies (4) - Summer No Marae",
        xaxis_title="Hour of Day",
        yaxis_title="SOC (%)",
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1
        ),
        template="plotly_white"
    )

    fig.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(range=[0, 100], gridcolor='lightgray', showgrid=True)

    add_peak_hour_shading(fig)

    output_file = OUTPUT_DIR / "soc_comparison_all_strategies_summer_no_marae.png"
    fig.write_image(output_file, width=1400, height=600, scale=2)
    print(f"  ✓ Saved: {output_file}")

    return fig


# ============================================================================
# VISUALIZATION 5: Grid Import/Export for All 12 Scenarios
# ============================================================================

def create_grid_import_export_all_scenarios():
    """
    Create separate grid import/export visualizations for all 12 scenarios
    """
    print(f"\nCreating Grid Import/Export visualizations for all scenarios...")

    results_base = Path(__file__).parent.parent / "results"

    for scenario_id in SCENARIO_MAP.values():
        scenario_dir = results_base / scenario_id

        print(f"  Creating Import/Export graph for: {scenario_id}")

        try:
            grid_file = scenario_dir / "res_ext_grid" / "p_mw.csv"
            if not grid_file.exists():
                print(f"    ⚠ Grid file not found: {grid_file}")
                continue

            df_grid = pd.read_csv(grid_file, delimiter=';', index_col=0)
            grid_power = df_grid.sum(axis=1) * 1000

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=grid_power.index,
                y=grid_power,
                name='Grid Power',
                marker=dict(
                    color=['#FF6347' if val >= 0 else '#32CD32' for val in grid_power],
                    line=dict(width=1, color='white')
                ),
                showlegend=False,
                hovertemplate='Hour %{x}<br>Power: %{y:.2f} kW<extra></extra>'
            ))

            # Add legend dummy traces
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='#FF6347'),
                name='Import (Grid → Community)',
                showlegend=True
            ))

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='#32CD32'),
                name='Export (Community → Grid)',
                showlegend=True
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5)

            scenario_title = scenario_id.replace('_', ' ').title()

            fig.update_layout(
                title=f"Grid Import/Export - {scenario_title}",
                xaxis_title="Hour of Day",
                yaxis_title="Power (kW)",
                hovermode='x unified',
                height=600,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="Black",
                    borderwidth=1
                ),
                template="plotly_white",
                annotations=[
                    dict(
                        text="Positive = Import (Grid → Community) | Negative = Export (Community → Grid)",
                        xref="paper", yref="paper",
                        x=0.5, y=1.08, showarrow=False,
                        font=dict(size=11, color="gray")
                    )
                ]
            )

            fig.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
            fig.update_yaxes(gridcolor='lightgray', showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black')

            # Add peak hour shading
            add_peak_hour_shading(fig)

            output_file = OUTPUT_DIR / f"grid_import_export_{scenario_id}.png"
            fig.write_image(output_file, width=1200, height=600, scale=2)
            print(f"    ✓ Saved: {output_file}")

        except Exception as e:
            print(f"    ⚠ Error creating Import/Export graph for {scenario_id}: {e}")
            continue


# ============================================================================
# VISUALIZATION 5B: Battery Power Flow - All 16 Scenarios
# ============================================================================

def create_battery_power_all_scenarios():
    """
    Create individual battery power flow graphs for all 16 scenarios
    Similar to grid import/export graphs but for battery charge/discharge
    """
    print(f"\nCreating Battery Power Flow visualizations for all scenarios...")

    results_base = Path(__file__).parent.parent / "results"

    for scenario_id in SCENARIO_MAP.values():
        scenario_dir = results_base / scenario_id

        print(f"  Creating Battery Power Flow graph for: {scenario_id}")

        try:
            batt_file = scenario_dir / "res_storage" / "p_mw.csv"
            if not batt_file.exists():
                print(f"    ⚠ Battery file not found: {batt_file}")
                continue

            df_batt = pd.read_csv(batt_file, delimiter=';', index_col=0)
            batt_power = df_batt.sum(axis=1) * 1000  # Convert MW to kW

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=batt_power.index,
                y=batt_power,
                name='Battery Power',
                marker=dict(
                    color=['#32CD32' if val >= 0 else '#FF6347' for val in batt_power],
                    line=dict(width=1, color='white')
                ),
                showlegend=False,
                hovertemplate='Hour %{x}<br>Power: %{y:.2f} kW<extra></extra>'
            ))

            # Add legend dummy traces
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='#32CD32'),
                name='Charge (Battery ← Grid)',
                showlegend=True
            ))

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='#FF6347'),
                name='Discharge (Battery → Grid)',
                showlegend=True
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5)

            scenario_title = scenario_id.replace('_', ' ').title()

            fig.update_layout(
                title=f"Battery Power Flow - {scenario_title}",
                xaxis_title="Hour of Day",
                yaxis_title="Power (kW)",
                hovermode='x unified',
                height=600,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="Black",
                    borderwidth=1
                ),
                template="plotly_white",
                annotations=[
                    dict(
                        text="Positive = Charge (Battery ← Grid) | Negative = Discharge (Battery → Grid)",
                        xref="paper", yref="paper",
                        x=0.5, y=1.08, showarrow=False,
                        font=dict(size=11, color="gray")
                    )
                ]
            )

            fig.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
            fig.update_yaxes(gridcolor='lightgray', showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black')

            # Add peak hour shading
            add_peak_hour_shading(fig)

            output_file = OUTPUT_DIR / f"battery_power_{scenario_id}.png"
            fig.write_image(output_file, width=1200, height=600, scale=2)
            print(f"    ✓ Saved: {output_file}")

        except Exception as e:
            print(f"    ⚠ Error creating Battery Power Flow graph for {scenario_id}: {e}")
            continue


# ============================================================================
# VISUALIZATION 6: Average Electricity Cost - All 16 Scenarios
# ============================================================================

def create_average_cost_visualization():
    """
    Create bar chart showing average electricity cost for all 12 scenarios
    """
    print(f"\nCreating Average Cost visualization for all scenarios...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.cost_calculator import CostCalculator

    config_path = Path(__file__).parent.parent / "config" / "scenario_definitions.yaml"
    calculator = CostCalculator(config_path=config_path)

    print(f"  Using tariff structure:")
    print(f"    Peak hours: {calculator.peak_hours}")
    print(f"    Peak import: ${calculator.peak_import_price}/kWh")
    print(f"    Off-peak import: ${calculator.offpeak_import_price}/kWh")
    print(f"    Export: ${calculator.export_price}/kWh")

    results_base = Path(__file__).parent.parent / "results"
    scenario_costs = {}

    for scenario_id in SCENARIO_MAP.values():
        scenario_dir = results_base / scenario_id

        print(f"  Calculating costs for: {scenario_id}")

        try:
            cost_df, summary = calculator.analyze_scenario(scenario_dir)

            load_file = scenario_dir / "res_load" / "p_mw.csv"
            if not load_file.exists():
                print(f"    ⚠ Load file not found: {load_file}")
                continue

            df_load = pd.read_csv(load_file, delimiter=';', index_col=0)
            total_demand_kwh = df_load.sum().sum() * 1000

            total_cost = summary['total_cost']

            if total_demand_kwh > 0:
                avg_cost_per_kwh = total_cost / total_demand_kwh
            else:
                avg_cost_per_kwh = 0

            scenario_costs[scenario_id] = {
                'total_cost': total_cost,
                'total_demand_kwh': total_demand_kwh,
                'total_import_kwh': summary['total_import_kwh'],
                'total_export_kwh': summary['total_export_kwh'],
                'avg_cost_per_kwh': avg_cost_per_kwh,
                'import_cost': summary['total_import_cost'],
                'export_revenue': summary['total_export_revenue']
            }

            print(f"    Total Cost: ${total_cost:.2f}")
            print(f"    Avg Cost/kWh: ${avg_cost_per_kwh:.4f}")

        except Exception as e:
            print(f"    ⚠ Error calculating costs for {scenario_id}: {e}")
            continue

    fig = go.Figure()

    # Separate by control type
    local_scenarios = [s for s in scenario_costs.keys() if 'local' in s]
    tou_scenarios = [s for s in scenario_costs.keys() if 'tou' in s]
    optimized_scenarios = [s for s in scenario_costs.keys() if 'optimized' in s]

    # Add local bars
    for scenario_id in local_scenarios:
        data = scenario_costs[scenario_id]
        display_name = scenario_id.replace('local_', '').replace('_', ' ').title()

        fig.add_trace(go.Bar(
            x=[display_name],
            y=[data['avg_cost_per_kwh']],
            name='Local Control',
            marker_color='#32CD32',
            showlegend=(scenario_id == local_scenarios[0]),
            text=[f"${data['avg_cost_per_kwh']:.4f}"],
            textposition='outside',
            hovertemplate=(
                f"<b>{display_name}</b><br>" +
                f"Avg Cost: ${data['avg_cost_per_kwh']:.4f}/kWh<br>" +
                f"Total Cost: ${data['total_cost']:.2f}<br>" +
                f"Demand: {data['total_demand_kwh']:.1f} kWh<br>" +
                f"Import: {data['total_import_kwh']:.1f} kWh<br>" +
                f"Export: {data['total_export_kwh']:.1f} kWh<br>" +
                "<extra></extra>"
            )
        ))

    # Add TOU bars
    for scenario_id in tou_scenarios:
        data = scenario_costs[scenario_id]
        display_name = scenario_id.replace('tou_', '').replace('_', ' ').title()

        fig.add_trace(go.Bar(
            x=[display_name],
            y=[data['avg_cost_per_kwh']],
            name='TOU Control',
            marker_color='#FF8C00',
            showlegend=(scenario_id == tou_scenarios[0]),
            text=[f"${data['avg_cost_per_kwh']:.4f}"],
            textposition='outside',
            hovertemplate=(
                f"<b>{display_name}</b><br>" +
                f"Avg Cost: ${data['avg_cost_per_kwh']:.4f}/kWh<br>" +
                f"Total Cost: ${data['total_cost']:.2f}<br>" +
                f"Demand: {data['total_demand_kwh']:.1f} kWh<br>" +
                f"Import: {data['total_import_kwh']:.1f} kWh<br>" +
                f"Export: {data['total_export_kwh']:.1f} kWh<br>" +
                "<extra></extra>"
            )
        ))

    # Add optimized bars
    for scenario_id in optimized_scenarios:
        data = scenario_costs[scenario_id]
        display_name = scenario_id.replace('optimized_', '').replace('_', ' ').title()

        fig.add_trace(go.Bar(
            x=[display_name],
            y=[data['avg_cost_per_kwh']],
            name='Optimized Control',
            marker_color='#9370DB',
            showlegend=(scenario_id == optimized_scenarios[0]),
            text=[f"${data['avg_cost_per_kwh']:.4f}"],
            textposition='outside',
            hovertemplate=(
                f"<b>{display_name}</b><br>" +
                f"Avg Cost: ${data['avg_cost_per_kwh']:.4f}/kWh<br>" +
                f"Total Cost: ${data['total_cost']:.2f}<br>" +
                f"Demand: {data['total_demand_kwh']:.1f} kWh<br>" +
                f"Import: {data['total_import_kwh']:.1f} kWh<br>" +
                f"Export: {data['total_export_kwh']:.1f} kWh<br>" +
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Average Electricity Cost per kWh - All Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Average Cost ($/kWh)",
        height=600,
        barmode='group',
        legend=dict(
            title="Control Strategy",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        annotations=[
            dict(
                text=f"Tariff: Peak ${calculator.peak_import_price}/kWh | Off-Peak ${calculator.offpeak_import_price}/kWh | Export ${calculator.export_price}/kWh",
                xref="paper", yref="paper",
                x=0.5, y=-0.15, showarrow=False,
                font=dict(size=10, color="gray")
            )
        ]
    )

    fig.update_xaxes(tickangle=45, gridcolor='lightgray', showgrid=False)
    fig.update_yaxes(gridcolor='lightgray', showgrid=True, rangemode='normal')

    output_file = OUTPUT_DIR / "average_cost_per_kwh_all_scenarios.png"
    fig.write_image(output_file, width=1600, height=700, scale=2)
    print(f"  ✓ Saved: {output_file}")

    return fig


# ============================================================================
# VISUALIZATION 7-10: Voltage and Loading (All 12 Scenarios)
# ============================================================================

def create_voltage_stability_diagrams():
    """Create voltage stability diagrams for all 12 scenarios"""
    print("\nCreating voltage stability diagrams for all scenarios...")

    for scenario_id in ALL_SCENARIOS:
        print(f"  Processing {scenario_id}...")

        voltage_file = RESULTS_DIR / scenario_id / "res_bus" / "vm_pu.csv"

        if not voltage_file.exists():
            print(f"    ⚠ Voltage file not found: {voltage_file}")
            continue

        df_voltage = pd.read_csv(voltage_file, delimiter=';', index_col=0)

        fig = go.Figure()

        bus_color = '#1f77b4'
        for idx, bus_name in enumerate(df_voltage.columns):
            fig.add_trace(go.Scatter(
                x=df_voltage.index,
                y=df_voltage[bus_name],
                name=bus_name,
                mode='lines',
                line=dict(width=1.5, color=bus_color),
                showlegend=(idx == 0),
                hovertemplate=f'{bus_name}<br>Hour %{{x}}<br>Voltage: %{{y:.3f}} p.u.<extra></extra>'
            ))

        fig.add_hline(y=1.05, line=dict(color='red', dash='dash', width=2), annotation_text="Upper Limit (1.05 p.u.)", annotation_position="right")
        fig.add_hline(y=0.95, line=dict(color='red', dash='dash', width=2), annotation_text="Lower Limit (0.95 p.u.)", annotation_position="right")
        fig.add_hline(y=1.0, line=dict(color='gray', dash='dot', width=1), annotation_text="Nominal (1.0 p.u.)", annotation_position="right")

        fig.update_layout(
            title=f"Voltage Stability - {scenario_id.replace('_', ' ').title()}",
            xaxis_title="Hour of Day",
            yaxis_title="Voltage (p.u.)",
            hovermode='x unified',
            height=600,
            width=1400,
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255, 255, 255, 0.8)"),
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True, range=[0.90, 1.10])
        )

        output_file = OUTPUT_DIR / f"voltage_stability_{scenario_id}.png"
        fig.write_image(output_file, width=1400, height=600, scale=2)
        print(f"  ✓ Saved: {output_file}")


def create_line_trafo_loading_diagrams():
    """Create line and transformer loading diagrams for all 12 scenarios"""
    print("\nCreating line and transformer loading diagrams for all scenarios...")

    for scenario_id in ALL_SCENARIOS:
        print(f"  Processing {scenario_id}...")

        line_file = RESULTS_DIR / scenario_id / "res_line" / "loading_percent.csv"
        trafo_file = RESULTS_DIR / scenario_id / "res_trafo" / "loading_percent.csv"

        if not line_file.exists():
            print(f"    ⚠ Line loading file not found: {line_file}")
            continue

        df_line = pd.read_csv(line_file, delimiter=';', index_col=0)

        has_trafo = trafo_file.exists()
        if has_trafo:
            df_trafo = pd.read_csv(trafo_file, delimiter=';', index_col=0)

        fig = go.Figure()

        line_color = '#1f77b4'
        for idx, line_name in enumerate(df_line.columns):
            fig.add_trace(go.Scatter(
                x=df_line.index,
                y=df_line[line_name],
                name=line_name,
                mode='lines',
                line=dict(width=1.5, color=line_color),
                legendgroup='lines',
                legendgrouptitle_text='Lines',
                showlegend=(idx == 0),
                hovertemplate=f'{line_name}<br>Hour %{{x}}<br>Loading: %{{y:.2f}}%<extra></extra>'
            ))

        if has_trafo:
            trafo_color = '#ff7f0e'
            for idx, trafo_name in enumerate(df_trafo.columns):
                fig.add_trace(go.Scatter(
                    x=df_trafo.index,
                    y=df_trafo[trafo_name],
                    name=trafo_name,
                    mode='lines',
                    line=dict(width=2.5, color=trafo_color, dash='dash'),
                    legendgroup='trafos',
                    legendgrouptitle_text='Transformers',
                    showlegend=(idx == 0),
                    hovertemplate=f'{trafo_name}<br>Hour %{{x}}<br>Loading: %{{y:.2f}}%<extra></extra>'
                ))

        fig.add_hline(y=100, line=dict(color='red', dash='dot', width=2), annotation_text="100% Loading", annotation_position="right")

        fig.update_layout(
            title=f"Line and Transformer Loading - {scenario_id.replace('_', ' ').title()}",
            xaxis_title="Hour of Day",
            yaxis_title="Loading (%)",
            hovermode='x unified',
            height=600,
            width=1400,
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255, 255, 255, 0.8)", tracegroupgap=10),
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True, rangemode='normal')
        )

        output_file = OUTPUT_DIR / f"line_trafo_loading_{scenario_id}.png"
        fig.write_image(output_file, width=1400, height=600, scale=2)
        print(f"  ✓ Saved: {output_file}")


def create_voltage_stability_heatmaps():
    """Create voltage stability heatmaps for all 12 scenarios"""
    print("\nCreating voltage stability heatmaps for all scenarios...")

    for scenario_id in ALL_SCENARIOS:
        print(f"  Processing {scenario_id}...")

        voltage_file = RESULTS_DIR / scenario_id / "res_bus" / "vm_pu.csv"

        if not voltage_file.exists():
            print(f"    ⚠ Voltage file not found: {voltage_file}")
            continue

        df_voltage = pd.read_csv(voltage_file, delimiter=';', index_col=0)
        df_voltage_T = df_voltage.T

        fig = go.Figure(data=go.Heatmap(
            z=df_voltage_T.values,
            x=df_voltage_T.columns,
            y=df_voltage_T.index,
            colorscale=[
                [0.0, '#FF0000'], [0.25, '#FFA500'], [0.40, '#FFFF00'],
                [0.50, '#90EE90'], [0.60, '#00FF00'], [0.75, '#FFA500'],
                [1.0, '#FF0000']
            ],
            zmid=1.0,
            zmin=0.95,
            zmax=1.05,
            colorbar=dict(title="Voltage (p.u.)", tickmode="linear", tick0=0.95, dtick=0.025),
            hovertemplate='Bus: %{y}<br>Hour: %{x}<br>Voltage: %{z:.3f} p.u.<extra></extra>'
        ))

        fig.update_layout(
            title=f"Voltage Stability Heatmap - {scenario_id.replace('_', ' ').title()}",
            xaxis_title="Hour of Day",
            yaxis_title="Bus",
            height=max(400, len(df_voltage_T.index) * 25),
            width=1400,
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True, autorange='reversed')
        )

        output_file = OUTPUT_DIR / f"voltage_stability_heatmap_{scenario_id}.png"
        fig.write_image(output_file, width=1400, height=max(400, len(df_voltage_T.index) * 25), scale=2)
        print(f"  ✓ Saved: {output_file}")


def create_line_trafo_loading_heatmaps():
    """Create line and transformer loading heatmaps for all 12 scenarios"""
    print("\nCreating line and transformer loading heatmaps for all scenarios...")

    for scenario_id in ALL_SCENARIOS:
        print(f"  Processing {scenario_id}...")

        line_file = RESULTS_DIR / scenario_id / "res_line" / "loading_percent.csv"
        trafo_file = RESULTS_DIR / scenario_id / "res_trafo" / "loading_percent.csv"

        if not line_file.exists():
            print(f"    ⚠ Line loading file not found: {line_file}")
            continue

        df_line = pd.read_csv(line_file, delimiter=';', index_col=0)

        has_trafo = trafo_file.exists()
        if has_trafo:
            df_trafo = pd.read_csv(trafo_file, delimiter=';', index_col=0)
            df_combined = pd.concat([df_line, df_trafo], axis=1)
        else:
            df_combined = df_line

        df_combined_T = df_combined.T

        fig = go.Figure(data=go.Heatmap(
            z=df_combined_T.values,
            x=df_combined_T.columns,
            y=df_combined_T.index,
            colorscale=[
                [0.0, '#00FF00'], [0.20, '#90EE90'], [0.40, '#FFFF99'],
                [0.60, '#FFFF00'], [0.70, '#FFD700'], [0.80, '#FFA500'],
                [0.90, '#FF6347'], [1.0, '#FF0000']
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(title="Loading (%)", tickmode="linear", tick0=0, dtick=10),
            hovertemplate='Element: %{y}<br>Hour: %{x}<br>Loading: %{z:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title=f"Line and Transformer Loading Heatmap - {scenario_id.replace('_', ' ').title()}",
            xaxis_title="Hour of Day",
            yaxis_title="Line / Transformer",
            height=max(400, len(df_combined_T.index) * 20),
            width=1400,
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True, autorange='reversed')
        )

        output_file = OUTPUT_DIR / f"line_trafo_loading_heatmap_{scenario_id}.png"
        fig.write_image(output_file, width=1400, height=max(400, len(df_combined_T.index) * 20), scale=2)
        print(f"  ✓ Saved: {output_file}")
# 11. CONTROL STRATEGY COMPARISONS
# ============================================================================

def create_control_strategy_comparisons():
    """
    Create comparison plots grouped by control strategy
    Compares Local, TOU, Optimized  strategies
    """
    print("\nCreating control strategy comparison visualizations...")

    # Group scenarios by control strategy
    local_scenarios = [s for s in ALL_SCENARIOS if 'local' in s]
    tou_scenarios = [s for s in ALL_SCENARIOS if 'tou' in s]
    optimized_scenarios = [s for s in ALL_SCENARIOS if 'optimized' in s]

    # ========================================================================
    # Strategy Comparison: Summer No Marae (All Three Strategies)
    # ========================================================================
    print("  Creating three-way strategy comparison (Summer No Marae)...")

    scenario_quartet = {
        'local_summer_no_marae': 'Local Control',
        'tou_summer_no_marae': 'TOU Control',
        'optimized_summer_no_marae': 'Optimized Control'
    }

    # Load data for all four strategies
    strategy_data = {}
    for scenario_id, display_name in scenario_quartet.items():
        scenario_dir = RESULTS_DIR / scenario_id

        try:
            # Load grid power
            grid_file = scenario_dir / "res_ext_grid" / "p_mw.csv"
            df_grid = pd.read_csv(grid_file, delimiter=';', index_col=0)
            grid_power = df_grid.sum(axis=1) * 1000

            # Load SOC
            soc_file = scenario_dir / "res_storage" / "soc_percent.csv"
            df_soc = pd.read_csv(soc_file, delimiter=';', index_col=0)

            # Load battery power
            batt_file = scenario_dir / "res_storage" / "p_mw.csv"
            df_batt = pd.read_csv(batt_file, delimiter=';', index_col=0)
            batt_power = df_batt.sum(axis=1) * 1000

            strategy_data[scenario_id] = {
                'display_name': display_name,
                'grid_power': grid_power,
                'soc': df_soc,
                'battery_power': batt_power
            }

        except Exception as e:
            print(f"    Error loading {scenario_id}: {e}")
            continue

    if len(strategy_data) >= 3:  # Allow 3 or 4 strategies
        colors = {
            'local_summer_no_marae': '#32CD32',         # Green
            'tou_summer_no_marae': '#FF8C00',           # Orange  
            'optimized_summer_no_marae': '#9370DB',     # Purple
        }

        # ====================================================================
        # Graph 1: Grid Power Comparison
        # ====================================================================
        fig_grid = go.Figure()

        for scenario_id, data_dict in strategy_data.items():
            grid_power = data_dict['grid_power']
            fig_grid.add_trace(go.Bar(
                x=grid_power.index,
                y=grid_power,
                name=data_dict['display_name'],
                marker=dict(
                    color=['#FF6347' if val >= 0 else '#32CD32' for val in grid_power],
                    line=dict(width=1, color='white')
                )
            ))

        # Add legend dummy traces for import/export indicators
        fig_grid.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#FF6347'),
            name='Import (Grid → Community)',
            showlegend=True
        ))

        fig_grid.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#32CD32'),
            name='Export (Community → Grid)',
            showlegend=True
        ))

        fig_grid.add_hline(y=0, line=dict(color='gray', dash='dash', width=1))
        add_peak_hour_shading(fig_grid)

        fig_grid.update_layout(
            title="Grid Import/Export Comparison: Summer No Marae",
            xaxis_title="Hour of Day",
            yaxis_title="Power (kW)",
            hovermode='x unified',
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            annotations=[
                dict(
                    text="Positive = Import (Grid → Community) | Negative = Export (Community → Grid)",
                    xref="paper", yref="paper",
                    x=0.5, y=1.08, showarrow=False,
                    font=dict(size=11, color="gray")
                )
            ]
        )

        fig_grid.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
        fig_grid.update_yaxes(gridcolor='lightgray', showgrid=True)

        output_file_grid = OUTPUT_DIR / "strategy_comparison_grid_power_summer_no_marae.png"
        fig_grid.write_image(output_file_grid, width=1200, height=600, scale=2)
        print(f"  ✓ Saved: {output_file_grid}")

        # ====================================================================
        # Graph 2: Battery SOC Comparison
        # ====================================================================
        fig_soc = go.Figure()

        for scenario_id, data_dict in strategy_data.items():
            soc_col = data_dict['soc'].columns[0]
            fig_soc.add_trace(go.Scatter(
                x=data_dict['soc'].index,
                y=data_dict['soc'][soc_col],
                name=data_dict['display_name'],
                line=dict(color=colors[scenario_id], width=2.5),
                mode='lines+markers',
                marker=dict(size=4)
            ))

        add_peak_hour_shading(fig_soc)

        fig_soc.update_layout(
            title="Battery State of Charge Comparison: Summer No Marae",
            xaxis_title="Hour of Day",
            yaxis_title="SOC (%)",
            hovermode='x unified',
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        fig_soc.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
        fig_soc.update_yaxes(range=[0, 100], gridcolor='lightgray', showgrid=True)

        output_file_soc = OUTPUT_DIR / "strategy_comparison_soc_summer_no_marae.png"
        fig_soc.write_image(output_file_soc, width=1200, height=600, scale=2)
        print(f"  ✓ Saved: {output_file_soc}")

        # ====================================================================
        # Graph 3: Battery Power Flow Comparison
        # ====================================================================
        fig_batt = go.Figure()

        for scenario_id, data_dict in strategy_data.items():
            batt_power = data_dict['battery_power']
            fig_batt.add_trace(go.Bar(
                x=batt_power.index,
                y=batt_power,
                name=data_dict['display_name'],
                marker=dict(
                    color=['#32CD32' if val >= 0 else '#FF6347' for val in batt_power],
                    line=dict(width=1, color='white')
                )
            ))

        # Add legend dummy traces for charge/discharge indicators
        fig_batt.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#32CD32'),
            name='Charge (Battery ← Grid)',
            showlegend=True
        ))

        fig_batt.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#FF6347'),
            name='Discharge (Battery → Grid)',
            showlegend=True
        ))

        fig_batt.add_hline(y=0, line=dict(color='gray', dash='dash', width=1))
        add_peak_hour_shading(fig_batt)

        fig_batt.update_layout(
            title="Battery Power Flow Comparison: Summer No Marae",
            xaxis_title="Hour of Day",
            yaxis_title="Power (kW)",
            hovermode='x unified',
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            annotations=[
                dict(
                    text="Positive = Charge (Battery ← Grid) | Negative = Discharge (Battery → Grid)",
                    xref="paper", yref="paper",
                    x=0.5, y=1.08, showarrow=False,
                    font=dict(size=11, color="gray")
                )
            ]
        )

        fig_batt.update_xaxes(dtick=2, gridcolor='lightgray', showgrid=True)
        fig_batt.update_yaxes(gridcolor='lightgray', showgrid=True)

        output_file_batt = OUTPUT_DIR / "strategy_comparison_battery_power_summer_no_marae.png"
        fig_batt.write_image(output_file_batt, width=1200, height=600, scale=2)
        print(f"  ✓ Saved: {output_file_batt}")

        # ====================================================================
        # Graph 4: Cost Comparison
        # ====================================================================
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.cost_calculator import CostCalculator

        config_path = Path(__file__).parent.parent / "config" / "scenario_definitions.yaml"
        calculator = CostCalculator(config_path=config_path)

        cost_values = []
        cost_labels = []
        cost_colors_list = []

        for scenario_id, data_dict in strategy_data.items():
            scenario_dir = RESULTS_DIR / scenario_id
            try:
                _, summary = calculator.analyze_scenario(scenario_dir)
                cost_values.append(summary['total_cost'])
                cost_labels.append(data_dict['display_name'])
                cost_colors_list.append(colors[scenario_id])
            except:
                pass

        if cost_values:
            fig_cost = go.Figure()

            fig_cost.add_trace(go.Bar(
                x=cost_labels,
                y=cost_values,
                marker_color=cost_colors_list,
                text=[f'${v:.2f}' for v in cost_values],
                textposition='outside'
            ))

            fig_cost.update_layout(
                title="Cost Comparison: Summer No Marae",
                xaxis_title="Control Strategy",
                yaxis_title="Total Cost ($)",
                height=600,
                template="plotly_white",
                showlegend=False
            )

            fig_cost.update_yaxes(gridcolor='lightgray', showgrid=True)

            output_file_cost = OUTPUT_DIR / "strategy_comparison_cost_summer_no_marae.png"
            fig_cost.write_image(output_file_cost, width=1200, height=600, scale=2)
            print(f"  ✓ Saved: {output_file_cost}")


# ============================================================================
# 12. SEASONAL COMPARISON (WITHIN SAME STRATEGY)
# ============================================================================

def create_seasonal_comparisons():
    """
    Create seasonal comparison plots for each control strategy
    Compares winter vs summer within same strategy
    """
    print("\nCreating seasonal comparison visualizations...")

    strategies = ['local', 'tou', 'optimized']
    
    for strategy in strategies:
        print(f"  Creating seasonal comparison for {strategy} control...")

        # Get winter and summer scenarios
        winter_scenario = f"{strategy}_winter_no_marae"
        summer_scenario = f"{strategy}_summer_no_marae"

        winter_dir = RESULTS_DIR / winter_scenario
        summer_dir = RESULTS_DIR / summer_scenario

        if not winter_dir.exists() or not summer_dir.exists():
            print(f"    ⚠ Missing data for {strategy} seasonal comparison")
            continue

        try:
            # Load winter data
            winter_grid = pd.read_csv(winter_dir / "res_ext_grid" / "p_mw.csv", delimiter=';', index_col=0).sum(axis=1) * 1000
            winter_soc = pd.read_csv(winter_dir / "res_storage" / "soc_percent.csv", delimiter=';', index_col=0)
            winter_pv = pd.read_csv(winter_dir / "res_sgen" / "p_mw.csv", delimiter=';', index_col=0).sum(axis=1) * 1000
            winter_load = pd.read_csv(winter_dir / "res_load" / "p_mw.csv", delimiter=';', index_col=0).sum(axis=1) * 1000

            # Load summer data
            summer_grid = pd.read_csv(summer_dir / "res_ext_grid" / "p_mw.csv", delimiter=';', index_col=0).sum(axis=1) * 1000
            summer_soc = pd.read_csv(summer_dir / "res_storage" / "soc_percent.csv", delimiter=';', index_col=0)
            summer_pv = pd.read_csv(summer_dir / "res_sgen" / "p_mw.csv", delimiter=';', index_col=0).sum(axis=1) * 1000
            summer_load = pd.read_csv(summer_dir / "res_load" / "p_mw.csv", delimiter=';', index_col=0).sum(axis=1) * 1000

            # Create figure
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'PV Generation (Winter vs Summer)',
                    'Total Load (Winter vs Summer)',
                    'Battery SOC (Winter vs Summer)',
                    'Grid Power (Winter vs Summer)'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.12
            )

            winter_color = '#1E90FF'  # Blue
            summer_color = '#FF8C00'  # Orange

            # Plot 1: PV Generation
            fig.add_trace(go.Scatter(
                x=winter_pv.index, y=winter_pv,
                name='Winter', line=dict(color=winter_color, width=2.5), mode='lines'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=summer_pv.index, y=summer_pv,
                name='Summer', line=dict(color=summer_color, width=2.5), mode='lines'
            ), row=1, col=1)

            # Plot 2: Load
            fig.add_trace(go.Scatter(
                x=winter_load.index, y=winter_load,
                name='Winter', line=dict(color=winter_color, width=2.5), mode='lines', showlegend=False
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=summer_load.index, y=summer_load,
                name='Summer', line=dict(color=summer_color, width=2.5), mode='lines', showlegend=False
            ), row=1, col=2)

            # Plot 3: SOC (first battery)
            winter_soc_col = winter_soc.columns[0]
            summer_soc_col = summer_soc.columns[0]
            
            fig.add_trace(go.Scatter(
                x=winter_soc.index, y=winter_soc[winter_soc_col],
                name='Winter', line=dict(color=winter_color, width=2.5), mode='lines+markers', showlegend=False
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=summer_soc.index, y=summer_soc[summer_soc_col],
                name='Summer', line=dict(color=summer_color, width=2.5), mode='lines+markers', showlegend=False
            ), row=2, col=1)

            add_peak_hour_shading(fig, row=2, col=1)

            # Plot 4: Grid Power
            fig.add_trace(go.Scatter(
                x=winter_grid.index, y=winter_grid,
                name='Winter', line=dict(color=winter_color, width=2.5), mode='lines', showlegend=False
            ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=summer_grid.index, y=summer_grid,
                name='Summer', line=dict(color=summer_color, width=2.5), mode='lines', showlegend=False
            ), row=2, col=2)

            fig.add_hline(y=0, line=dict(color='gray', dash='dash', width=1), row=2, col=2)

            # Update axes
            fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
            fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
            fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
            fig.update_yaxes(title_text="SOC (%)", row=2, col=1)
            fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

            fig.update_layout(
                title=f"Seasonal Comparison: {strategy.upper()} Control (Winter vs Summer)",
                height=800,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )

            output_file = OUTPUT_DIR / f"seasonal_comparison_{strategy}.png"
            fig.write_image(output_file, width=1400, height=800, scale=2)
            print(f"    ✓ Saved: {output_file}")

        except Exception as e:
            print(f"    ⚠ Error creating seasonal comparison for {strategy}: {e}")




# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("MICROGRID VISUALIZATION GENERATOR")
    print("="*70)

    # ========================================================================
    # PART 1: INDIVIDUAL SCENARIO VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("PART 1: INDIVIDUAL SCENARIO VISUALIZATIONS")
    print("="*70)

    print("\n[1] Creating Individual Load Profile Visualizations")
    print("-" * 70)
    for sheet in SHEET_NAMES:
        create_load_profiles_visualization(sheet)

    print("\n[2] Creating Aggregated PV vs Load Visualization")
    print("-" * 70)
    create_aggregated_pv_load_visualization()

    print("\n[3] Creating SOC Profile Visualizations (All 16 Scenarios)")
    print("-" * 70)
    create_soc_profiles_all_scenarios()

    print("\n[4] Creating Grid Import/Export Visualizations (All 16 Scenarios)")
    print("-" * 70)
    create_grid_import_export_all_scenarios()

    print("\n[5] Creating Battery Power Flow Visualizations (All 16 Scenarios)")
    print("-" * 70)
    create_battery_power_all_scenarios()

    print("\n[6] Creating Voltage Stability Diagrams (All 16 Scenarios)")
    print("-" * 70)
    create_voltage_stability_diagrams()

    print("\n[7] Creating Line and Transformer Loading Diagrams (All 16 Scenarios)")
    print("-" * 70)
    create_line_trafo_loading_diagrams()

    print("\n[8] Creating Voltage Stability Heatmaps (All 16 Scenarios)")
    print("-" * 70)
    create_voltage_stability_heatmaps()

    print("\n[9] Creating Line and Transformer Loading Heatmaps (All 16 Scenarios)")
    print("-" * 70)
    create_line_trafo_loading_heatmaps()

    # ========================================================================
    # PART 2: COMPARISON VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("PART 2: COMPARISON VISUALIZATIONS")
    print("="*70)

    print("\n[10] Creating SOC Comparison: All Strategies (Summer No Marae)")
    print("-" * 70)
    create_soc_comparison_all_strategies_summer()

    print("\n[11] Creating Average Cost per kWh Visualization")
    print("-" * 70)
    create_average_cost_visualization()

    print("\n[12] Creating Control Strategy Comparisons")
    print("-" * 70)
    create_control_strategy_comparisons()

    print("\n[13] Creating Seasonal Comparisons")
    print("-" * 70)
    create_seasonal_comparisons()

    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE")
    print(f"All visualizations saved to: {OUTPUT_DIR.absolute()}")
    print("="*70)