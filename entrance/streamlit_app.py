# streamlit_app.py - Interactive interface using new architecture
"""
Makarika Microgrid Interactive Dashboard
=========================================

This Streamlit web application provides an interactive interface for simulating
and analyzing microgrid performance with different battery control strategies,
seasonal variations, and operational scenarios.

What it does:
-------------
- Interactive web dashboard for microgrid simulation
- Visual scenario selection (control strategy, season, events)
- Real-time power flow simulation with pandapower
- Interactive network topology map with live results
- Time-series visualizations of voltages, power flows, and battery states
- Economic analysis with cost breakdowns and tariff visualization
- Export results to CSV for further analysis


How to use:
-----------
1. Launch the dashboard:
   
   streamlit run streamlit_app.py

2. Configure your scenario in the sidebar:
   - Select battery control strategy
   - Choose season (summer/winter)
   - Toggle marae event on/off
3. Click "Run Simulation" to execute
4. Explore results in different tabs:


Interface Structure:
--------------------
- Sidebar: Configuration and cost summary
- Main tabs: Network, Voltages, Power, Battery, Economics
- Each tab shows relevant plots and data tables

Technical Details:
------------------
- Built with Streamlit for web interface
- Uses Plotly for interactive charts
- Folium for network mapping
- Integrates with simulation architecture:
  * NetworkManager: Network topology
  * ScenarioManager: Load/generation profiles
  * GridSimulator: Power flow simulation
  * CostCalculator: Economic analysis

Dependencies:
-------------
- streamlit >= 1.29.0
- plotly >= 5.18.0
- folium >= 0.15.0
- streamlit-folium
- pandas, pandapower

Notes:
------
- First run initializes all components (may take a few seconds)
- Results cached in session state for fast re-analysis
- Network topology loaded from predefined structure
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running from entrance/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from streamlit_folium import st_folium  # pip install streamlit-folium

# Import architecture components
from models.network_manager import NetworkManager
from scenarios.scenario_manager import ScenarioManager
from simulation.simulator import GridSimulator
from visualization.network_map import NetworkVisualizer
from utils.cost_calculator import CostCalculator

st.set_page_config(page_title="Makarika Microgrid", layout="wide", page_icon="⚡")

# Initialize session state
if 'initialized' not in st.session_state:
    # Use absolute paths relative to the script's parent directory (project root)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "venv/data/"
    config_path = project_root / "config/scenario_definitions.yaml"

    with st.spinner("Initializing system..."):
        network_mgr = NetworkManager(data_dir)
        network_mgr.build_base_network()

        scenario_mgr = ScenarioManager(config_path, network_mgr, data_dir)
        simulator = GridSimulator(network_mgr, scenario_mgr, data_dir)
        visualizer = NetworkVisualizer(network_mgr)
        cost_calculator = CostCalculator(config_path=config_path)

    st.session_state.network_mgr = network_mgr
    st.session_state.scenario_mgr = scenario_mgr
    st.session_state.simulator = simulator
    st.session_state.visualizer = visualizer
    st.session_state.cost_calculator = cost_calculator
    st.session_state.initialized = True
    st.session_state.results = None
    st.session_state.cost_summary = None

# Header
st.title(" Makarika Microgrid Simulator")
st.markdown("Interactive tool for analyzing distributed energy resources")

# Sidebar - Configuration
st.sidebar.header("🎛️ Configuration")

# ============================================================================
# Scenario selection with structured dropdowns
# ============================================================================
st.sidebar.subheader("Scenario Parameters")

# Control Strategy Selection
control_strategy = st.sidebar.selectbox(
    "Control Strategy",
    ["local", "tou", "optimized"],
    format_func=lambda x: "Local Battery Control" if x == "local" else ("Time-of-Use Control" if x == "tou" else "Optimized Battery Control"),
    help="Local: Each battery optimizes its local bus independently\nToU: Batteries charge during off-peak and discharge during peak hours\nOptimized: Uses pre-computed optimal battery schedules"
)

# Season Selection
season = st.sidebar.selectbox(
    "Season",
    ["winter", "summer"],
    format_func=lambda x: x.capitalize(),
    help="Affects PV capacity factors and load profiles"
)

# Marae Event Selection
marae_event = st.sidebar.selectbox(
    "Marae Event",
    [False, True],
    format_func=lambda x: "With Marae Event" if x else "No Marae Event",
    help="Marae events increase community load consumption"
)

# Construct scenario ID from selections
marae_suffix = "marae" if marae_event else "no_marae"
selected_scenario = f"{control_strategy}_{season}_{marae_suffix}"

# Get scenario details
scenarios = st.session_state.scenario_mgr.list_scenarios()
scenario_dict = {s['id']: s for s in scenarios}

if selected_scenario in scenario_dict:
    scenario_info = scenario_dict[selected_scenario]
    st.sidebar.success(f"**Selected:** {scenario_info['name']}")
    with st.sidebar.expander("ℹ️ Scenario Details"):
        st.write(scenario_info['description'])
else:
    st.sidebar.error(f"⚠️ Scenario '{selected_scenario}' not found in configuration!")
    st.sidebar.info("Available scenarios:\n" + "\n".join([f"- {s['id']}" for s in scenarios]))

# Network Modifications
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Network Modifications")

# Get current network
net = st.session_state.network_mgr.current_net

# PV Controls
# with st.sidebar.expander("☀️ PV Systems"):
#     pv_modifications = {}
#     pv_scales = {}

#     st.markdown("**Existing PV Systems:**")
#     for idx, pv in net.sgen.iterrows():
#         # Get bus name - pv['bus'] is the bus index (position in bus table)
#         bus_idx = pv['bus']
#         if bus_idx in net.bus.index:
#             bus_name = net.bus.at[bus_idx, 'name']
#         else:
#             bus_name = "Unknown"
#         pv_name = pv['name']

#         col1, col2 = st.columns([3, 1])
#         with col1:
#             enabled = st.checkbox(
#                 f"{pv_name} @ {bus_name}",
#                 value=pv['in_service'],
#                 key=f"pv_enable_{idx}"
#             )
#             pv_modifications[pv_name] = enabled

#         with col2:
#             st.caption(f"{pv['p_mw']*1000:.0f}kW")

#         # Add scaling slider
#         scale = st.slider(
#             f"Scale {pv_name}",
#             0.5, 3.0, 1.0, 0.1,
#             key=f"pv_scale_{idx}",
#             help="Scale PV capacity (1.0 = original, 2.0 = double capacity)"
#         )
#         pv_scales[pv_name] = scale

pv_modifications = {}
pv_scales = {}

    # st.markdown("---")
    # st.markdown("**Add New PV:**")
    # # COMMENTED OUT - PV addition feature not yet working
    # # Get LV buses with loads and filter out those that already have PV
    # # First, find all buses that have loads
    # buses_with_loads = set()
    # for idx, load in net.load.iterrows():
    #     if load['in_service']:
    #         bus_idx = load['bus']
    #         if bus_idx in net.bus.index:
    #             bus_name = net.bus.at[bus_idx, 'name']
    #             # Only consider LV buses
    #             if net.bus.at[bus_idx, 'vn_kv'] < 1.0:
    #                 buses_with_loads.add(bus_name)

    # # Find buses that already have PV systems
    # buses_with_pv = set()
    # for idx, pv in net.sgen.iterrows():
    #     bus_idx = pv['bus']
    #     if bus_idx in net.bus.index:
    #         bus_name = net.bus.at[bus_idx, 'name']
    #         buses_with_pv.add(bus_name)

    # # Filter: only buses with loads AND without existing PV
    # available_buses = sorted([bus for bus in buses_with_loads if bus not in buses_with_pv])

    # if not available_buses:
    #     st.info("ℹ️ All LV buses with loads already have PV systems. Use scaling to increase capacity.")
    # else:
    #     new_pv_bus = st.selectbox("Bus", available_buses, key="new_pv_bus")
    #     new_pv_capacity = st.number_input("Capacity (kW)", 5, 50, 15, 5, key="new_pv_capacity")

    #     if st.button("➕ Add PV"):
    #         try:
    #             pv_name = f"PV_{new_pv_bus}"
    #             st.session_state.network_mgr.add_pv_system(new_pv_bus, new_pv_capacity, name=pv_name)
    #             st.session_state.network_mgr.apply_scenario(selected_scenario)
    #             st.session_state.visualizer = NetworkVisualizer(st.session_state.network_mgr)
    #             st.success(f"Added {new_pv_capacity}kW PV at {new_pv_bus}!")
    #             st.rerun()
    #         except Exception as e:
    #             st.error(f"Error: {e}")

# Line Controls
with st.sidebar.expander("🔌 Line Status"):
    line_modifications = {}
    
    for idx, line in net.line.iterrows():
        from_bus_idx = line['from_bus']
        to_bus_idx = line['to_bus']

        if from_bus_idx in net.bus.index:
            from_bus = net.bus.at[from_bus_idx, 'name']
        else:
            from_bus = "Unknown"

        if to_bus_idx in net.bus.index:
            to_bus = net.bus.at[to_bus_idx, 'name']
        else:
            to_bus = "Unknown"

        connected = st.checkbox(
            f"{line['name']}: {from_bus} → {to_bus}",
            value=line['in_service'],
            key=f"line_{idx}"
        )
        line_modifications[line['name']] = connected

# Load Scaling
# with st.sidebar.expander("🏠 Load Scaling"):
#     load_scales = {}
#     
#     for idx, load in net.load.iterrows():
#         bus_idx = load['bus']
#         if bus_idx in net.bus.index:
#             bus_name = net.bus.at[bus_idx, 'name']
#         else:
#             bus_name = "Unknown"
#         scale = st.slider(
#             f"{load['name']} @ {bus_name}",
#             0.5, 2.0, 1.0, 0.1,
#             key=f"load_{idx}"
#         )
#         load_scales[load['name']] = scale

# Simulation Settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Simulation")

# Reset button
# if st.sidebar.button("🔄 Reset Network"):
#     st.session_state.network_mgr.reset_network()
#     st.session_state.visualizer = NetworkVisualizer(st.session_state.network_mgr)
#     st.success("Network reset!")
#     st.rerun()

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    
    with st.spinner("Running simulation..."):
        network_mgr = st.session_state.network_mgr

        # Store user-added PV systems before reset (not in base network)
        user_added_pv = []
        for pv_name, pv_info in network_mgr.component_registry.items():
            if pv_info.get('type') == 'pv' and pv_info.get('enabled'):
                # Check if this PV is NOT in the base network (i.e., user-added)
                if pv_name not in ['PV_bus_10', 'PV_bus_18']:  # Base PV systems
                    user_added_pv.append({
                        'bus': pv_info['bus'],
                        'capacity_kw': pv_info['capacity_kw'],
                        'name': pv_name
                    })

        # Apply network modifications
        network_mgr.reset_network()

        # Re-add user-added PV systems
        for pv_config in user_added_pv:
            try:
                network_mgr.add_pv_system(
                    pv_config['bus'],
                    pv_config['capacity_kw'],
                    name=pv_config['name']
                )
                print(f"✅ Re-added user PV: {pv_config['name']}")
            except Exception as e:
                print(f"⚠️ Could not re-add PV {pv_config['name']}: {e}")

        # Apply PV toggles
        for pv_name, enabled in pv_modifications.items():
            if pv_name in network_mgr.component_registry:
                network_mgr.toggle_pv_system(pv_name, enabled)

        # Apply PV scaling
        # for pv_name, scale in pv_scales.items():
        #     if pv_name in network_mgr.component_registry:
        #         network_mgr.scale_pv_system(pv_name, scale)

        # Apply line status
        for line_name, connected in line_modifications.items():
            if line_name in network_mgr.component_registry:
                if connected:
                    network_mgr.reconnect_line(line_name)
                else:
                    network_mgr.disconnect_line(line_name)
        
        # Apply load scaling
        # for load_name, scale in load_scales.items():
        #     if load_name in network_mgr.component_registry:
        #         network_mgr.scale_load(load_name, scale)
        
        # Update visualizer with modified network
        st.session_state.visualizer = NetworkVisualizer(network_mgr)
        
        # Run simulation
        try:
            results = st.session_state.simulator.run_scenario(
                selected_scenario,
                n_steps=24
            )
            st.session_state.results = results

            # Calculate costs for the scenario
            try:
                cost_df, cost_summary = st.session_state.cost_calculator.analyze_scenario(
                    results['output_path']
                )
                # Export cost analysis
                st.session_state.cost_calculator.export_cost_analysis(
                    cost_df,
                    cost_summary,
                    scenario_name=selected_scenario
                )
                st.session_state.cost_summary = cost_summary
            except Exception as cost_error:
                st.warning(f"⚠️ Cost calculation failed: {cost_error}")
                st.session_state.cost_summary = None

            st.success("✅ Simulation completed!")
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
            st.exception(e)
            st.session_state.results = None
            st.session_state.cost_summary = None

# Main content tabs
tab_map, tab_results, tab_stability, tab_costs = st.tabs([
    "🗺️ Network Map", "📊 Power Flows", "⚡ Stability", "💰 Costs & Energy"
])

with tab_map:
    st.header("Interactive Network Map")

    # Create and display map
    network_map = st.session_state.visualizer.create_interactive_map()

    # Display map in Streamlit
    st_folium(network_map, width=1200, height=600)
    
    # Network summary below map
    st.markdown("---")
    st.subheader("Current Network Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        active_pv = len(net.sgen[net.sgen.in_service])
        total_pv = len(net.sgen)
        st.metric("PV Systems", f"{active_pv}/{total_pv}")

        if active_pv < total_pv:
            disabled = [pv['name'] for _, pv in net.sgen.iterrows() if not pv['in_service']]
            st.caption(f"Disabled: {', '.join(disabled)}")

    with col2:
        connected_lines = len(net.line[net.line.in_service])
        total_lines = len(net.line)
        st.metric("Lines", f"{connected_lines}/{total_lines}")

        if connected_lines < total_lines:
            disconnected = [line['name'] for _, line in net.line.iterrows() if not line['in_service']]
            st.caption(f"Disconnected: {', '.join(disconnected)}")

    with col3:
        # Get rated capacity from component registry (not current output)
        total_pv_capacity = 0
        for pv_name, pv_info in st.session_state.network_mgr.component_registry.items():
            if pv_info.get('type') == 'pv' and pv_info.get('enabled', False):
                total_pv_capacity += pv_info.get('capacity_kw', 0)
        st.metric("PV Capacity", f"{total_pv_capacity:.0f} kW")

# Other tabs remain the same as before
with tab_results:
    if st.session_state.results:
        results = st.session_state.results
        net = st.session_state.network_mgr.current_net

        # Show which scenario results are being displayed
        st.header("📊 Power Flow Results")
        results_path = results['output_path']
        scenario_name = results_path.name

        # Get scenario display name
        scenario_info = next((s for s in scenarios if s['id'] == scenario_name), None)
        if scenario_info:
            st.info(f"**Showing results for:** {scenario_info['name']}\n\n")
        else:
            st.info(f"**Showing results for:** {scenario_name}\n\n")
              

        # Bus selection dropdown - only show buses with loads
        buses_with_loads = net.load['bus'].unique()
        bus_names_with_loads = [net.bus.at[bus_idx, 'name'] for bus_idx in buses_with_loads]
        bus_options = ["All Buses (Sum)"] + sorted(bus_names_with_loads)

        selected_bus = st.selectbox(
            "Select Bus for Detailed View",
            bus_options,
            help="Choose 'All Buses (Sum)' for aggregate view, or select a specific bus with loads"
        )

        # Load raw data from CSV files for per-bus analysis
        results_path = results['output_path']

        try:
            load_data = pd.read_csv(results_path / "res_load" / "p_mw.csv",
                                   index_col=0, delimiter=';')
            sgen_data = pd.read_csv(results_path / "res_sgen" / "p_mw.csv",
                                   index_col=0, delimiter=';')
            storage_data = pd.read_csv(results_path / "res_storage" / "p_mw.csv",
                                      index_col=0, delimiter=';')
            grid_data = pd.read_csv(results_path / "res_ext_grid" / "p_mw.csv",
                                   index_col=0, delimiter=';')
            soc_data = pd.read_csv(results_path / "res_storage" / "soc_percent.csv",
                                   index_col=0, delimiter=';')
        except Exception as e:
            st.error(f"Error loading detailed results: {e}")
            load_data = pd.DataFrame()
            sgen_data = pd.DataFrame()
            storage_data = pd.DataFrame()
            grid_data = pd.DataFrame()
            soc_data = pd.DataFrame()

        if selected_bus == "All Buses (Sum)":
            # Use aggregated data from results
            data = results['data']
            time = list(data['time_hours'])

            load_series = data['load']
            pv_series = data['pv_generation']
            battery_series = data['battery_power']
            grid_series = data['grid_power']
            soc_series = data['battery_soc']

            has_pv = True
            has_battery = True

        else:
            # Get bus-specific data
            time = list(results['data']['time_hours'])
            bus_idx = net.bus[net.bus['name'] == selected_bus].index[0]

            # Find loads at this bus
            bus_loads = net.load[net.load['bus'] == bus_idx]
            if not bus_loads.empty and not load_data.empty:
                load_series = pd.Series([0.0] * len(time), index=load_data.index)
                for _, load_row in bus_loads.iterrows():
                    load_name = load_row['name']
                    if load_name in load_data.columns:
                        load_series += load_data[load_name] * 1000  # Convert to kW
            else:
                load_series = pd.Series([0.0] * len(time))

            # Find PV at this bus
            bus_pv = net.sgen[net.sgen['bus'] == bus_idx]
            has_pv = not bus_pv.empty
            if has_pv and not sgen_data.empty:
                pv_series = pd.Series([0.0] * len(time), index=sgen_data.index)
                for _, pv_row in bus_pv.iterrows():
                    pv_name = pv_row['name']
                    if pv_name in sgen_data.columns:
                        pv_series += sgen_data[pv_name] * 1000  # Convert to kW
            else:
                pv_series = pd.Series([0.0] * len(time))

            # Find battery at this bus
            if hasattr(net, 'storage'):
                bus_battery = net.storage[net.storage['bus'] == bus_idx]
                has_battery = not bus_battery.empty
                if has_battery and not storage_data.empty:
                    battery_series = pd.Series([0.0] * len(time), index=storage_data.index)
                    soc_series = pd.Series([50.0] * len(time), index=soc_data.index)
                    for _, batt_row in bus_battery.iterrows():
                        batt_name = batt_row['name']
                        if batt_name in storage_data.columns:
                            battery_series += storage_data[batt_name] * 1000  # Convert to kW
                        if batt_name in soc_data.columns:
                            soc_series = soc_data[batt_name]
                else:
                    battery_series = pd.Series([0.0] * len(time))
                    soc_series = pd.Series([50.0] * len(time))
            else:
                has_battery = False
                battery_series = pd.Series([0.0] * len(time))
                soc_series = pd.Series([50.0] * len(time))

            # Grid power (always show for individual buses)
            grid_series = grid_data.iloc[:, 0] * 1000 if not grid_data.empty else pd.Series([0.0] * len(time))

        # Display appropriate graphs based on what's available
        if selected_bus != "All Buses (Sum)" and not has_pv and not has_battery:
            # Only show load
            st.info(f"ℹ️ Bus '{selected_bus}' has no PV system or battery. Showing load only.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=load_series, name='Load',
                                    line=dict(color='red', width=2)))

            fig.update_layout(
                title="Load Over Time",
                xaxis_title="Time (hours)",
                yaxis_title="Power (kW)",
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_bus != "All Buses (Sum)" and has_pv and not has_battery:
            # Show load and PV only
            st.info(f"ℹ️ Bus '{selected_bus}' has PV but no battery.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=load_series, name='Load',
                                    line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=time, y=pv_series, name='PV',
                                    line=dict(color='orange', width=2)))

            fig.update_layout(
                title="Load vs PV Generation",
                xaxis_title="Time (hours)",
                yaxis_title="Power (kW)",
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_bus != "All Buses (Sum)" and not has_pv and has_battery:
            # Show load, battery, and SOC
            st.info(f"ℹ️ Bus '{selected_bus}' has battery but no PV system.")

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Load Over Time', 'Battery Power', 'SOC Profile')
            )

            fig.add_trace(go.Scatter(x=time, y=load_series, name='Load',
                                    line=dict(color='red', width=2)), row=1, col=1)
            fig.add_trace(go.Bar(x=time, y=battery_series, name='Battery',
                                marker_color='blue'), row=1, col=2)
            fig.add_trace(go.Scatter(x=time, y=soc_series, name='SOC',
                                    line=dict(color='purple', width=3)), row=1, col=3)

            # Update y-axis titles
            fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
            fig.update_yaxes(title_text="Power (kW)<br><sub>+Discharge | -Charge</sub>", row=1, col=2)
            fig.update_yaxes(title_text="SOC (%)", row=1, col=3)
            fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
            fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
            fig.update_xaxes(title_text="Time (hours)", row=1, col=3)

            fig.update_layout(height=400, showlegend=True, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Full view with all components for "All Buses (Sum)"
            if selected_bus == "All Buses (Sum)":
                # Show all 4 graphs including grid exchange
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Load vs PV Generation', 'Battery Power',
                                  'Grid Exchange', 'SOC Profile')
                )

                fig.add_trace(go.Scatter(x=time, y=load_series, name='Load',
                                        line=dict(color='red', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=time, y=pv_series, name='PV',
                                        line=dict(color='orange', width=2)), row=1, col=1)
                fig.add_trace(go.Bar(x=time, y=battery_series, name='Battery',
                                    marker_color='blue'), row=1, col=2)
                fig.add_trace(go.Bar(x=time, y=grid_series, name='Grid',
                                    marker_color='green'), row=2, col=1)
                fig.add_trace(go.Scatter(x=time, y=soc_series, name='SOC',
                                        line=dict(color='purple', width=3)), row=2, col=2)

                # Update y-axis titles
                fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
                fig.update_yaxes(title_text="Power (kW)<br><sub>+Discharge | -Charge</sub>", row=1, col=2)
                fig.update_yaxes(title_text="Power (kW)<br><sub>+Import | -Export</sub>", row=2, col=1)
                fig.update_yaxes(title_text="SOC (%)", row=2, col=2)
                fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
                fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
                fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
                fig.update_xaxes(title_text="Time (hours)", row=2, col=2)

                fig.update_layout(height=700, showlegend=True, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Individual bus with both PV and battery - show 3 graphs (no grid)
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Load vs PV Generation', 'Battery Power', 'SOC Profile')
                )

                fig.add_trace(go.Scatter(x=time, y=load_series, name='Load',
                                        line=dict(color='red', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=time, y=pv_series, name='PV',
                                        line=dict(color='orange', width=2)), row=1, col=1)
                fig.add_trace(go.Bar(x=time, y=battery_series, name='Battery',
                                    marker_color='blue'), row=1, col=2)
                fig.add_trace(go.Scatter(x=time, y=soc_series, name='SOC',
                                        line=dict(color='purple', width=3)), row=1, col=3)

                # Update y-axis titles
                fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
                fig.update_yaxes(title_text="Power (kW)<br><sub>+Discharge | -Charge</sub>", row=1, col=2)
                fig.update_yaxes(title_text="SOC (%)", row=1, col=3)
                fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
                fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
                fig.update_xaxes(title_text="Time (hours)", row=1, col=3)

                fig.update_layout(height=400, showlegend=True, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        st.markdown("---")
        st.subheader("Key Metrics")

        # For "All Buses (Sum)" - show Net Exchange
        # For individual buses - show Battery Cycles instead
        if selected_bus == "All Buses (Sum)":
            # Show 4 columns including Net Exchange
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Load", f"{load_series.sum():.1f} kWh")
            with col2:
                if has_pv:
                    st.metric("PV Generation", f"{pv_series.sum():.1f} kWh")
                else:
                    st.metric("PV Generation", "N/A")
            with col3:
                grid_import = grid_series[grid_series > 0].sum()
                grid_export = abs(grid_series[grid_series < 0].sum())
                net_exchange = grid_import - grid_export
                st.metric("Net Exchange", f"{net_exchange:.1f} kWh")
            with col4:
                if has_pv and load_series.sum() > 0:
                    self_suff = min(100, (pv_series.sum() / load_series.sum()) * 100)
                    st.metric("Self-Sufficiency", f"{self_suff:.1f}%")
                else:
                    st.metric("Self-Sufficiency", "N/A")
        else:
            # Individual bus - show only 3 metrics (no Net Exchange)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Load", f"{load_series.sum():.1f} kWh")
            with col2:
                if has_pv:
                    st.metric("PV Generation", f"{pv_series.sum():.1f} kWh")
                else:
                    st.metric("PV Generation", "N/A")
            with col3:
                if has_pv and load_series.sum() > 0:
                    self_suff = min(100, (pv_series.sum() / load_series.sum()) * 100)
                    st.metric("Self-Sufficiency", f"{self_suff:.1f}%")
                else:
                    st.metric("Self-Sufficiency", "N/A")
    else:
        st.info("👈 Run a simulation to see power flow results")

with tab_stability:
    if st.session_state.results:
        st.header("Voltage Stability Analysis")
        results_path = st.session_state.results['output_path']
        net = st.session_state.network_mgr.current_net

        try:
            # Load voltage data
            voltage_file = results_path / "res_bus" / "vm_pu.csv"
            voltage_df = pd.read_csv(voltage_file, delimiter=';', index_col=0)

            time = list(st.session_state.results['data']['time_hours'])

            # Visualization options
            viz_type = st.radio(
                "Visualization Type",
                ["Line Chart (All Buses)", "Heatmap"],
                horizontal=True
            )

            if viz_type == "Line Chart (All Buses)":
                st.subheader("Voltage Profiles Over Time")

                # Bus selection - single dropdown with "All Buses" as default
                all_buses = sorted(voltage_df.columns.tolist())
                bus_options = ["All Buses"] + all_buses

                selected_bus_filter = st.selectbox(
                    "Select Bus",
                    bus_options,
                    index=0,
                    help="Choose 'All Buses' to show all buses, or select a specific bus"
                )

                # Determine which buses to display
                if selected_bus_filter == "All Buses":
                    bus_filter = voltage_df.columns.tolist()
                else:
                    bus_filter = [selected_bus_filter]

                # Create line chart
                fig = go.Figure()

                for bus_name in bus_filter:
                    if bus_name in voltage_df.columns:
                        # Color code by voltage level and type
                        if 'trafo' in bus_name.lower():
                            # Transformers - distinctive styling
                            color = 'purple'
                            width = 3
                            opacity = 1.0
                            dash = 'solid'
                        elif 'lv' in bus_name.lower():
                            color = 'lightblue'
                            width = 1.5
                            opacity = 0.7
                            dash = 'solid'
                        else:
                            # MV buses
                            color = 'orange'
                            width = 1.5
                            opacity = 0.7
                            dash = 'solid'

                        fig.add_trace(go.Scatter(
                            x=time,
                            y=voltage_df[bus_name],
                            mode='lines',
                            name=bus_name,
                            line=dict(color=color, width=width, dash=dash),
                            opacity=opacity,
                            showlegend=False,  # Remove legend
                            hovertemplate=f'<b>{bus_name}</b><br>Voltage: %{{y:.4f}} pu<br>Time: %{{x}}h<extra></extra>'
                        ))

                # Add threshold lines
                fig.add_hline(y=1.05, line_dash="dash", line_color="red",
                             annotation_text="Upper Limit (1.05 pu)", annotation_position="right")
                fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                             annotation_text="Lower Limit (0.95 pu)", annotation_position="right")
                fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                             annotation_text="Nominal (1.0 pu)", annotation_position="right")

                fig.update_layout(
                    title="Bus Voltage Stability Over Time",
                    xaxis_title="Time (hours)",
                    yaxis_title="Voltage (pu)",
                    height=600,
                    hovermode='x unified',
                    yaxis=dict(range=[0.93, 1.07]),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add color legend as text
                st.markdown("""
                **Color Legend:**
                - 🟣 **Purple (thick):** Transformers
                - 🔵 **Light Blue:** LV Buses (0.4 kV)
                - 🟠 **Orange:** MV Buses (11 kV)
                """)

            elif viz_type == "Heatmap":
                st.subheader("Voltage Deviation Heatmap")

                # Calculate deviation from nominal (1.0 pu)
                deviation_df = (voltage_df - 1.0) * 100  # Convert to percentage

                fig = go.Figure(data=go.Heatmap(
                    z=deviation_df.T.values,
                    x=time,
                    y=deviation_df.columns,
                    colorscale='RdYlGn_r',  # Red for high deviation, green for nominal
                    zmid=0,  # Center colorscale at 0%
                    colorbar=dict(title="Deviation (%)"),
                    hovertemplate='Bus: %{y}<br>Time: %{x}h<br>Deviation: %{z:.2f}%<extra></extra>'
                ))

                fig.update_layout(
                    title="Voltage Deviation from Nominal (1.0 pu)",
                    xaxis_title="Time (hours)",
                    yaxis_title="Bus Name",
                    height=800
                )

                st.plotly_chart(fig, use_container_width=True)

            # Voltage violations summary
            st.markdown("---")
            st.subheader("Voltage Violations")

            col1, col2, col3, col4 = st.columns(4)

            # Calculate violations
            overvoltage = (voltage_df > 1.05).sum().sum()
            undervoltage = (voltage_df < 0.95).sum().sum()
            max_voltage = voltage_df.max().max()
            min_voltage = voltage_df.min().min()

            with col1:
                st.metric("Overvoltage Events", int(overvoltage),
                         delta="Above 1.05 pu" if overvoltage > 0 else None,
                         delta_color="inverse")
            with col2:
                st.metric("Undervoltage Events", int(undervoltage),
                         delta="Below 0.95 pu" if undervoltage > 0 else None,
                         delta_color="inverse")
            with col3:
                st.metric("Max Voltage", f"{max_voltage:.4f} pu",
                         delta=f"+{(max_voltage-1.0)*100:.2f}%" if max_voltage > 1.0 else None)
            with col4:
                st.metric("Min Voltage", f"{min_voltage:.4f} pu",
                         delta=f"{(min_voltage-1.0)*100:.2f}%" if min_voltage < 1.0 else None)

            # Show violation details if any
            if overvoltage > 0 or undervoltage > 0:
                with st.expander("⚠️ View Violation Details"):
                    for bus_name in voltage_df.columns:
                        bus_violations = ((voltage_df[bus_name] > 1.05) | (voltage_df[bus_name] < 0.95))
                        if bus_violations.any():
                            violation_times = voltage_df.index[bus_violations].tolist()
                            violation_values = voltage_df.loc[bus_violations, bus_name].tolist()
                            st.write(f"**{bus_name}:** {len(violation_times)} violations")
                            for t, v in zip(violation_times, violation_values):
                                st.write(f"  - Hour {t}: {v:.4f} pu")

        except Exception as e:
            st.error(f"Error loading voltage data: {e}")
            st.exception(e)

        # Transformer and Line Loading
        st.markdown("---")
        st.header("Transformer & Line Loading Analysis")
        
        try:
            # Load transformer and line loading data
            trafo_file = results_path / "res_trafo" / "loading_percent.csv"
            line_file = results_path / "res_line" / "loading_percent.csv"
            
            trafo_df = pd.read_csv(trafo_file, delimiter=';', index_col=0)
            line_df = pd.read_csv(line_file, delimiter=';', index_col=0)
            
            time = list(st.session_state.results['data']['time_hours'])
            
            # Selection options
            st.subheader("Loading Profiles Over Time")
            
            # Build selection options
            trafo_list = sorted(trafo_df.columns.tolist())
            line_list = sorted(line_df.columns.tolist())
            
            # Create grouped options
            selection_options = ["All Transformers & Lines"] + \
                               [f"🔶 {t}" for t in trafo_list] + \
                               [f"🔷 {l}" for l in line_list]
            
            selected_element = st.selectbox(
                "Select Element",
                selection_options,
                index=0,
                help="Choose 'All' to show everything, or select a specific transformer or line"
            )
            
            # Create loading chart
            fig = go.Figure()
            
            if selected_element == "All Transformers & Lines":
                # Plot all transformers
                for trafo_name in trafo_df.columns:
                    fig.add_trace(go.Scatter(
                        x=time,
                        y=trafo_df[trafo_name],
                        mode='lines',
                        name=trafo_name,
                        line=dict(color='#ff7f0e', width=3),  # Orange for transformers
                        opacity=0.8,
                        legendgroup='transformers',
                        legendgrouptitle_text='Transformers',
                        hovertemplate=f'<b>{trafo_name}</b><br>Loading: %{{y:.2f}}%<br>Time: %{{x}}h<extra></extra>'
                    ))
                
                # Plot all lines
                for line_name in line_df.columns:
                    fig.add_trace(go.Scatter(
                        x=time,
                        y=line_df[line_name],
                        mode='lines',
                        name=line_name,
                        line=dict(color='#1f77b4', width=1.5),  # Blue for lines
                        opacity=0.5,
                        legendgroup='lines',
                        legendgrouptitle_text='Lines',
                        hovertemplate=f'<b>{line_name}</b><br>Loading: %{{y:.2f}}%<br>Time: %{{x}}h<extra></extra>'
                    ))
                
                show_legend = False  # Too many elements for legend
                
            elif selected_element.startswith("🔶"):
                # Single transformer selected
                trafo_name = selected_element[2:].strip()
                fig.add_trace(go.Scatter(
                    x=time,
                    y=trafo_df[trafo_name],
                    mode='lines+markers',
                    name=trafo_name,
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{trafo_name}</b><br>Loading: %{{y:.2f}}%<br>Time: %{{x}}h<extra></extra>'
                ))
                show_legend = True
                
            else:
                # Single line selected
                line_name = selected_element[2:].strip()
                fig.add_trace(go.Scatter(
                    x=time,
                    y=line_df[line_name],
                    mode='lines+markers',
                    name=line_name,
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{line_name}</b><br>Loading: %{{y:.2f}}%<br>Time: %{{x}}h<extra></extra>'
                ))
                show_legend = True
            
            # Add 100% loading threshold
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                         annotation_text="100% Loading Limit", annotation_position="right")
            
            fig.update_layout(
                title="Transformer & Line Loading Over Time",
                xaxis_title="Time (hours)",
                yaxis_title="Loading (%)",
                height=600,
                hovermode='x unified',
                yaxis=dict(range=[0, max(trafo_df.max().max(), line_df.max().max()) * 1.1]),
                showlegend=show_legend
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add color legend
            st.markdown("""
            **Color Legend:**
            - 🔶 **Orange (thick):** Transformers
            - 🔷 **Blue (thin):** Lines
            """)
            
            # Loading statistics
            st.markdown("---")
            st.subheader("Loading Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate statistics
            max_trafo_loading = trafo_df.max().max()
            max_line_loading = line_df.max().max()
            avg_trafo_loading = trafo_df.values.mean()
            avg_line_loading = line_df.values.mean()
            
            # Count overloading events
            trafo_overload = (trafo_df > 100).sum().sum()
            line_overload = (line_df > 100).sum().sum()
            
            with col1:
                st.metric("Max Transformer Loading", f"{max_trafo_loading:.1f}%",
                         delta="Overloaded!" if max_trafo_loading > 100 else "Within limits",
                         delta_color="inverse" if max_trafo_loading > 100 else "normal")
            
            with col2:
                st.metric("Max Line Loading", f"{max_line_loading:.1f}%",
                         delta="Overloaded!" if max_line_loading > 100 else "Within limits",
                         delta_color="inverse" if max_line_loading > 100 else "normal")
            
            with col3:
                st.metric("Avg Transformer Loading", f"{avg_trafo_loading:.1f}%")
            
            with col4:
                st.metric("Avg Line Loading", f"{avg_line_loading:.1f}%")
            
            # Show overloading details if any
            if trafo_overload > 0 or line_overload > 0:
                with st.expander("⚠️ View Overloading Details"):
                    if trafo_overload > 0:
                        st.write("**Transformer Overloading:**")
                        for trafo_name in trafo_df.columns:
                            overloads = trafo_df[trafo_name] > 100
                            if overloads.any():
                                overload_times = trafo_df.index[overloads].tolist()
                                overload_values = trafo_df.loc[overloads, trafo_name].tolist()
                                st.write(f"**{trafo_name}:** {len(overload_times)} overload events")
                                for t, v in zip(overload_times, overload_values):
                                    st.write(f"  - Hour {t}: {v:.1f}%")
                    
                    if line_overload > 0:
                        st.write("**Line Overloading:**")
                        for line_name in line_df.columns:
                            overloads = line_df[line_name] > 100
                            if overloads.any():
                                overload_times = line_df.index[overloads].tolist()
                                overload_values = line_df.loc[overloads, line_name].tolist()
                                st.write(f"**{line_name}:** {len(overload_times)} overload events")
                                for t, v in zip(overload_times, overload_values):
                                    st.write(f"  - Hour {t}: {v:.1f}%")
            
        except Exception as e:
            st.error(f"Error loading transformer/line data: {e}")
            st.exception(e)

    else:
        st.info("👈 Run a simulation to see voltage stability analysis")

with tab_costs:
    if st.session_state.results:
        results = st.session_state.results
        data = results['data']

        st.header("💰 Costs & Energy Balance")

        # Cost breakdown section
        if st.session_state.cost_summary:
            st.subheader("Cost Analysis")

            cost_col1, cost_col2, cost_col3 = st.columns(3)

            with cost_col1:
                import_cost = st.session_state.cost_summary['total_import_cost']
                st.metric("Import Cost", f"${import_cost:.2f}")
            with cost_col2:
                export_revenue = st.session_state.cost_summary['total_export_revenue']
                st.metric("Export Revenue", f"${export_revenue:.2f}")
            with cost_col3:
                net_cost = st.session_state.cost_summary['total_cost']
                st.metric("Net Daily Cost", f"${net_cost:.2f}")

            # Detailed cost breakdown
            st.markdown("---")
            st.subheader("Detailed Cost Breakdown")

            cost_details_col1, cost_details_col2 = st.columns(2)

            with cost_details_col1:
                st.markdown("**Import Costs**")
                peak_cost = st.session_state.cost_summary['peak_import_cost']
                offpeak_cost = st.session_state.cost_summary['offpeak_import_cost']
                st.write(f"Peak Hours: ${peak_cost:.2f}")
                st.write(f"Off-Peak Hours: ${offpeak_cost:.2f}")
                st.write(f"**Total Import: ${import_cost:.2f}**")

            with cost_details_col2:
                st.markdown("**Energy Flows**")
                total_import = st.session_state.cost_summary['total_import_kwh']
                total_export = st.session_state.cost_summary['total_export_kwh']
                net_exchange = st.session_state.cost_summary['net_exchange_kwh']
                st.write(f"Grid Import: {total_import:.1f} kWh")
                st.write(f"Grid Export: {total_export:.1f} kWh")
                st.write(f"**Net Exchange: {net_exchange:.1f} kWh**")

        # Hourly tariff information
        st.markdown("---")
        st.subheader("Tariff Structure")

        tariff_col1, tariff_col2, tariff_col3 = st.columns(3)

        with tariff_col1:
            st.markdown("**Peak Import**")
            peak_price = st.session_state.cost_calculator.peak_import_price
            peak_hours = st.session_state.cost_calculator.peak_hours
            st.write(f"${peak_price:.2f}/kWh")
            st.caption(f"Hours: {', '.join(map(str, peak_hours))}")

        with tariff_col2:
            st.markdown("**Off-Peak Import**")
            offpeak_price = st.session_state.cost_calculator.offpeak_import_price
            st.write(f"${offpeak_price:.2f}/kWh")
            st.caption("All other hours")

        with tariff_col3:
            st.markdown("**Export Revenue**")
            export_price = st.session_state.cost_calculator.export_price
            st.write(f"${export_price:.2f}/kWh")
            st.caption("All hours")

        # Hourly cost visualization
        st.markdown("---")
        st.subheader("Hourly Cost Breakdown")

        # Load cost timestep data
        results_path = results['output_path']
        cost_timestep_file = Path("results/costs") / results_path.name / "cost_timesteps.csv"

        if cost_timestep_file.exists():
            cost_timesteps = pd.read_csv(cost_timestep_file)

            # Create hourly cost chart (single plot without tariff)
            fig = go.Figure()

            # Plot import cost and export revenue
            fig.add_trace(
                go.Bar(
                    x=cost_timesteps['hour'],
                    y=cost_timesteps['import_cost'],
                    name='Import Cost',
                    marker_color='red',
                    hovertemplate='Hour %{x}<br>Import Cost: $%{y:.3f}<extra></extra>'
                )
            )

            fig.add_trace(
                go.Bar(
                    x=cost_timesteps['hour'],
                    y=-cost_timesteps['export_revenue'],  # Negative to show as revenue
                    name='Export Revenue',
                    marker_color='green',
                    hovertemplate='Hour %{x}<br>Export Revenue: $%{y:.3f}<extra></extra>'
                )
            )

            # Add peak hour shading
            peak_hours = st.session_state.cost_calculator.peak_hours
            for peak_hour in peak_hours:
                fig.add_vrect(
                    x0=peak_hour - 0.5, x1=peak_hour + 0.5,
                    fillcolor="orange", opacity=0.1,
                    layer="below", line_width=0
                )

            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Cost ($)")

            fig.update_layout(
                height=400,
                showlegend=True,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Energy Balance section
        st.markdown("---")
        st.subheader("Energy Balance")

        # Calculate battery charge/discharge first
        battery_charge = abs(data['battery_power'][data['battery_power'] < 0].sum())
        battery_discharge = data['battery_power'][data['battery_power'] > 0].sum()
        
        # Load energy balance data from CSV to get losses
        energy_balance_file = results['output_path'] / 'energy_balance.csv'
        if energy_balance_file.exists():
            energy_balance = pd.read_csv(energy_balance_file)
            # Sum losses (convert from MWh to kWh)
            line_losses = (energy_balance['line_loss_mw'].sum() * 1000)
            trafo_losses = (energy_balance['trafo_loss_mw'].sum() * 1000)
            # Check if battery_loss_mw column exists (new column)
            if 'battery_loss_mw' in energy_balance.columns:
                battery_losses = (energy_balance['battery_loss_mw'].sum() * 1000)
            else:
                # Fallback for old simulations
                battery_losses = battery_charge - (battery_discharge * 0.9)
                if battery_losses < 0:
                    battery_losses = 0
            network_losses = line_losses + trafo_losses
        else:
            line_losses = 0
            trafo_losses = 0
            battery_losses = 0
            network_losses = 0

        energy_data = {
            'Component': ['Load Demand', 'PV Generation', 'Battery Discharge',
                        'Battery Charge', 'Grid Import', 'Grid Export',
                        'Battery Losses', 'Network Losses (Lines)', 'Network Losses (Transformers)'],
            'Energy (kWh)': [
                data['load'].sum(),
                data['pv_generation'].sum(),
                battery_discharge,
                battery_charge,
                data['grid_power'][data['grid_power'] > 0].sum(),
                abs(data['grid_power'][data['grid_power'] < 0].sum()),
                battery_losses,
                line_losses,
                trafo_losses
            ],
            'Type': ['Consumption', 'Generation', 'Generation', 'Consumption',
                    'Generation', 'Consumption', 'Consumption', 'Consumption', 'Consumption']
        }
        df_energy = pd.DataFrame(energy_data)
        
        st.dataframe(df_energy[['Component', 'Energy (kWh)']], hide_index=True, use_container_width=True)
        

    else:
        st.info("👈 Run a simulation to see cost and energy analysis")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("""
**Microgrid Simulator**  
Interactive geographic network visualization  
Built with Streamlit + Folium + pandapower
""")