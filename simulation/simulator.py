# simulation/simulator.py
"""
Grid Simulator - Time-Series Power Flow Engine
===============================================

Orchestrates time-series simulations by coordinating the network, scenarios,
and battery controllers to analyze microgrid performance over 24-hour periods.

What it does:
-------------
- Runs time-series power flow simulations using pandapower
- Coordinates network topology, load/PV profiles, and controllers
- Saves detailed results (voltages, power flows, SOC, grid imports)
- Handles controller execution at each timestep
- Manages output writing to CSV files
- Processes and packages simulation results

Simulation Flow:
----------------
1. Apply scenario configuration (loads, PV, controllers)
2. Initialize output writer with result variables
3. Run timestep loop:
   - Update load/PV from profiles
   - Execute controller decisions
   - Run power flow (pandapower.runpp)
   - Log results
4. Package results and save to disk



Saved Results:
--------------
Results saved to: results/{scenario_name}/

Structure:
- res_bus/vm_pu.csv: Bus voltages (per-unit)
- res_load/p_mw.csv: Load power consumption
- res_sgen/p_mw.csv: PV generation
- res_storage/p_mw.csv: Battery power (±)
- res_storage/soc_percent.csv: Battery state-of-charge
- res_ext_grid/p_mw.csv: Grid import/export
- res_line/loading_percent.csv: Line loading
- res_trafo/loading_percent.csv: Transformer loading

Output Format:
--------------
CSV files with:
- Rows: Timesteps (0 to n_steps-1)
- Columns: Elements (buses, loads, generators, batteries)
- Delimiter: Semicolon (;)



Integration:
------------
Used by:
- main.py: Single scenario execution
- streamlit_app.py: Interactive dashboard simulations
- run_stochastic_monte_carlo.py: Batch stochastic analysis
- Sensitivity analysis scripts

Dependencies:
-------------
- pandapower: Power flow solver
- NetworkManager: Provides network topology
- ScenarioManager: Configures scenarios
- Controllers: Execute battery strategies
"""

import pandapower as pp
from pandapower.timeseries import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
from pathlib import Path
import pandas as pd

class GridSimulator:
    """Main simulation engine - replaces your run_simulation.py"""
    
    def __init__(self, network_manager, scenario_manager, data_dir):
        self.network_manager = network_manager
        self.scenario_manager = scenario_manager
        self.data_dir = Path(data_dir)
        self.results = {}
    
    def run_scenario(self, scenario_name, n_steps=24, results_dir='results'):
        """
        Run a complete scenario simulation
        
        Args:
            scenario_name: Name of scenario to run
            n_steps: Number of timesteps
            results_dir: Directory to save results
        
        Returns:
            dict with simulation results
        """
        print(f"\n{'='*60}")
        print(f" STARTING SIMULATION: {scenario_name}")
        print(f"{'='*60}\n")

        # Apply scenario configuration
        net = self.scenario_manager.apply_scenario(scenario_name)

        # Debug: Print active controllers
        print(f"\n🔍 DEBUG: Active controllers in network:")
        for idx, ctrl in enumerate(net.controller.object):
            ctrl_type = type(ctrl).__name__
            print(f"  Controller {idx}: {ctrl_type}")
        print()
        
        # Setup output directory
        results_path = Path(results_dir) / scenario_name.replace(' ', '_')
        results_path.mkdir(parents=True, exist_ok=True)

        print(f"🔍 DEBUG: Results will be saved to: {results_path.absolute()}\n")

        # Configure output writer
        ow = OutputWriter(net, time_steps=range(n_steps),
                         output_path=results_path, 
                         output_file_type='.csv')
        
        # Log relevant variables
        ow.log_variable('res_load', 'p_mw')
        ow.log_variable('res_sgen', 'p_mw')
        ow.log_variable('res_storage', 'p_mw')
        ow.log_variable('res_storage', 'soc_percent')
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_line', 'loading_percent')
        ow.log_variable('res_line', 'pl_mw')
        ow.log_variable('res_trafo', 'loading_percent')
        ow.log_variable('res_trafo', 'pl_mw')
        ow.log_variable('res_ext_grid', 'p_mw')
        
        print(f"📊 Output configured to: {results_path}")
        print(f"⏱️  Timesteps: {n_steps}")
        print(f"🎛️  Active controllers: {len(net.controller)}\n")
        
        # Ensure all dtypes are correct before simulation (critical for numba)
        self._ensure_network_dtypes(net)

        # Run time-series simulation
        try:
            run_timeseries(net, time_steps=range(n_steps),
                          run_function=pp.runpp,
                          continue_on_divergence=True,
                          verbose=False)
            
            print(f"\n✅ Simulation completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Simulation failed: {e}")
            raise
        
        # Replace element indices with meaningful names in all result files FIRST
        self._replace_indices_with_names(net, results_path)

        #verify energy balance 
        self._verify_energy_balance(net, results_path, n_steps)

        # Then save battery SOC histories (after column names are updated)
        self._save_battery_histories(net, results_path)

        # Load and process results
        results = self._load_results(results_path, n_steps)
        
        # Store results
        self.results[scenario_name] = {
            'data': results,
            'network_state': net,
            'output_path': results_path
        }
        
        # Print summary
        self._print_summary(scenario_name, results)
        
        return self.results[scenario_name]
    
    def _ensure_network_dtypes(self, net):
        """Ensure all network DataFrames have correct dtypes for numba compilation"""
        import numpy as np

        # Fix bus reference columns (must be int64)
        bus_columns = {
            'load': ['bus'],
            'sgen': ['bus'],
            'storage': ['bus'],
            'line': ['from_bus', 'to_bus'],
            'trafo': ['hv_bus', 'lv_bus'],
            'ext_grid': ['bus']
        }

        for element_type, cols in bus_columns.items():
            if hasattr(net, element_type):
                df = getattr(net, element_type)
                if len(df) > 0:
                    for col in cols:
                        if col in df.columns:
                            # Remove any NaN values first
                            invalid_rows = df[df[col].isna()].index
                            if len(invalid_rows) > 0:
                                print(f"⚠️ Removing {len(invalid_rows)} invalid rows from {element_type}")
                                df.drop(invalid_rows, inplace=True)
                            # Convert to int64
                            if len(df) > 0:
                                df[col] = df[col].astype(np.int64)

        # Fix float columns (must be float64)
        float_columns = {
            'load': ['p_mw', 'q_mvar'],
            'sgen': ['p_mw', 'q_mvar'],
            'storage': ['p_mw', 'max_e_mwh', 'soc_percent', 'min_e_mwh'],
            'line': ['length_km'],
            'bus': ['vn_kv']
        }

        for element_type, cols in float_columns.items():
            if hasattr(net, element_type):
                df = getattr(net, element_type)
                if len(df) > 0:
                    for col in cols:
                        if col in df.columns:
                            df[col] = df[col].astype(np.float64)

        # Fix boolean columns
        bool_columns = {
            'load': ['in_service'],
            'sgen': ['in_service'],
            'storage': ['in_service'],
            'line': ['in_service'],
            'trafo': ['in_service'],
            'ext_grid': ['in_service']
        }

        for element_type, cols in bool_columns.items():
            if hasattr(net, element_type):
                df = getattr(net, element_type)
                if len(df) > 0:
                    for col in cols:
                        if col in df.columns:
                            df[col] = df[col].astype(bool)

        print("✅ Network types validated and fixed")

    def _save_battery_histories(self, net, results_path):
        """Save SOC histories from battery controllers to res_storage folder"""
        battery_controllers = [c for c in net.controller.object if hasattr(c, 'soc_history')]
        centralized_controllers = [c for c in net.controller.object if hasattr(c, 'battery1_soc_history')]

        if not battery_controllers and not centralized_controllers:
            return

        # Read the existing soc_percent.csv file (already has column names after _replace_indices_with_names)
        soc_file = results_path / 'res_storage' / 'soc_percent.csv'

        try:
            # Load the SOC data (columns should now be battery names like "Battery_bus10")
            soc_df = pd.read_csv(soc_file, index_col=0, delimiter=';')

            # Update with actual SOC data from standard controllers using battery names
            for controller in battery_controllers:
                if hasattr(controller, 'soc_history') and len(controller.soc_history) > 0:
                    # Get the battery name from the network
                    storage_idx = controller.storage_idx if hasattr(controller, 'storage_idx') else 0
                    battery_name = net.storage.at[storage_idx, 'name']

                    # Update the column with actual SOC history using the battery name
                    if battery_name in soc_df.columns and len(controller.soc_history) == len(soc_df):
                        soc_df[battery_name] = controller.soc_history
                        print(f"   ✓ Updated SOC for {battery_name}")
                    else:
                        print(f"   ⚠️ Could not find column '{battery_name}' in SOC file or length mismatch")

            # Update with actual SOC data from centralized controllers
            for controller in centralized_controllers:
                # Update Battery 1
                if hasattr(controller, 'battery1_soc_history') and len(controller.battery1_soc_history) > 0:
                    battery1_name = net.storage.at[controller.battery1_idx, 'name']
                    if battery1_name in soc_df.columns and len(controller.battery1_soc_history) == len(soc_df):
                        soc_df[battery1_name] = controller.battery1_soc_history
                        print(f"   ✓ Updated SOC for {battery1_name} (centralized)")
                    else:
                        print(f"   ⚠️ Could not find column '{battery1_name}' in SOC file or length mismatch")
                
                # Update Battery 2
                if hasattr(controller, 'battery2_soc_history') and len(controller.battery2_soc_history) > 0:
                    battery2_name = net.storage.at[controller.battery2_idx, 'name']
                    if battery2_name in soc_df.columns and len(controller.battery2_soc_history) == len(soc_df):
                        soc_df[battery2_name] = controller.battery2_soc_history
                        print(f"   ✓ Updated SOC for {battery2_name} (centralized)")
                    else:
                        print(f"   ⚠️ Could not find column '{battery2_name}' in SOC file or length mismatch")

            # Save back to the file
            soc_df.to_csv(soc_file, sep=';')
            print(f"✅ Updated SOC data in {soc_file}")

        except Exception as e:
            print(f"⚠️ Warning: Could not update SOC file: {e}")

        # Also save individual battery histories to res_storage for reference
        storage_dir = results_path / 'res_storage'
        for controller in battery_controllers:
            if hasattr(controller, 'save_soc_history'):
                bus_name = getattr(controller, 'bus_name', 'unknown')
                filename = storage_dir / f'soc_history_{bus_name}.csv'
                controller.save_soc_history(str(filename))

    def _replace_indices_with_names(self, net, results_path):
        """Replace element integer indices with meaningful names in all result CSV files"""
        results_path = Path(results_path)

        # Create mappings for all element types
        mappings = {
            'res_bus': {idx: row['name'] for idx, row in net.bus.iterrows()},
            'res_load': {idx: row['name'] for idx, row in net.load.iterrows()},
            'res_sgen': {idx: row['name'] for idx, row in net.sgen.iterrows()},
            'res_line': {idx: row['name'] for idx, row in net.line.iterrows()},
            'res_trafo': {idx: row['name'] for idx, row in net.trafo.iterrows()},
        }

        # Add storage mapping if storage exists
        if hasattr(net, 'storage') and len(net.storage) > 0:
            mappings['res_storage'] = {idx: row['name'] for idx, row in net.storage.iterrows()}

        # Process each result folder
        for folder_name, element_map in mappings.items():
            folder_path = results_path / folder_name

            if not folder_path.exists():
                continue

            # Process all CSV files in the folder
            for csv_file in folder_path.glob('*.csv'):
                try:
                    # Read the CSV
                    df = pd.read_csv(csv_file, delimiter=';', index_col=0)

                    # Rename columns using element names
                    new_columns = []
                    for col in df.columns:
                        try:
                            elem_idx = int(col)
                            new_columns.append(element_map.get(elem_idx, col))
                        except ValueError:
                            # If column name is not an integer, keep it as is
                            new_columns.append(col)

                    df.columns = new_columns

                    # Save back to file
                    df.to_csv(csv_file, sep=';')

                except Exception as e:
                    print(f"⚠️ Warning: Could not rename columns in {csv_file.name}: {e}")

        print(f"✅ Replaced element indices with names in result files")

    def _verify_energy_balance(self, net, results_path, n_steps):

        try:
            # Load result files
            grid_file = results_path / "res_ext_grid" / "p_mw.csv"
            load_file = results_path / "res_load" / "p_mw.csv"
            sgen_file = results_path / "res_sgen" / "p_mw.csv"
            storage_file = results_path / "res_storage" / "p_mw.csv"
            line_loss_file = results_path / "res_line" / "pl_mw.csv"
            trafo_loss_file = results_path / "res_trafo" / "pl_mw.csv"
            
            # Load data
            df_grid = pd.read_csv(grid_file, delimiter=';', index_col=0)
            df_load = pd.read_csv(load_file, delimiter=';', index_col=0)
            df_sgen = pd.read_csv(sgen_file, delimiter=';', index_col=0) if sgen_file.exists() else pd.DataFrame()
            df_storage = pd.read_csv(storage_file, delimiter=';', index_col=0) if storage_file.exists() else pd.DataFrame()
            df_line_loss = pd.read_csv(line_loss_file, delimiter=';', index_col=0) if line_loss_file.exists() else pd.DataFrame()
            df_trafo_loss = pd.read_csv(trafo_loss_file, delimiter=';', index_col=0) if trafo_loss_file.exists() else pd.DataFrame()
            
            # Calculate balance for each timestep
            balance_data = []
            
            for t in range(n_steps):
                # Sources (MW) - actual power generation
                grid_power = df_grid.iloc[t].sum() if t < len(df_grid) else 0
                pv_power = df_sgen.iloc[t].sum() if not df_sgen.empty and t < len(df_sgen) else 0
                
                # Battery (MW) - positive = discharging (generation), negative = charging (consumption)
                battery_power = df_storage.iloc[t].sum() if not df_storage.empty and t < len(df_storage) else 0
                
                # Calculate battery losses (charging/discharging inefficiency)
                # Loss occurs on both charge and discharge (assume 95% one-way efficiency)
                battery_loss = 0
                if not df_storage.empty and t < len(df_storage):
                    for col in df_storage.columns:
                        p = df_storage.iloc[t][col]
                        if p < 0:  # Charging - energy consumed but not fully stored
                            battery_loss += abs(p) * 0.05  # 5% loss on charge
                        elif p > 0:  # Discharging - stored energy lost during discharge
                            battery_loss += p * 0.0526  # 5% loss on discharge (1/0.95 - 1)
                
                # Total generation (sources providing power to the system)
                # Battery discharging is generation, battery charging is consumption
                battery_generation = max(0, battery_power)  # Only positive values
                battery_consumption = abs(min(0, battery_power))  # Only negative values (made positive)
                total_generation = grid_power + pv_power + battery_generation
                
                # Consumption (MW)
                load_power = df_load.iloc[t].sum() if t < len(df_load) else 0
                line_losses = df_line_loss.iloc[t].sum() if not df_line_loss.empty and t < len(df_line_loss) else 0
                trafo_losses = df_trafo_loss.iloc[t].sum() if not df_trafo_loss.empty and t < len(df_trafo_loss) else 0
                total_losses = line_losses + trafo_losses + battery_loss
                total_consumption = load_power + battery_consumption + total_losses
                
                # Imbalance
                imbalance = total_generation - total_consumption
                relative_error = abs(imbalance) / max(total_generation, total_consumption) * 100 if max(total_generation, total_consumption) > 0 else 0
                
                balance_data.append({
                    'timestep': t,
                    'grid_mw': grid_power,
                    'pv_mw': pv_power,
                    'battery_mw': battery_power,
                    'total_generation_mw': total_generation,
                    'load_mw': load_power,
                    'line_loss_mw': line_losses,
                    'trafo_loss_mw': trafo_losses,
                    'battery_loss_mw': battery_loss,
                    'total_losses_mw': total_losses,
                    'total_consumption_mw': total_consumption,
                    'imbalance_mw': imbalance,
                    'relative_error_pct': relative_error
                })
            
            # Save to CSV
            df_balance = pd.DataFrame(balance_data)
            balance_file = results_path / "energy_balance.csv"
            df_balance.to_csv(balance_file, index=False)
            
            # Print summary
            total_losses_kwh = df_balance['total_losses_mw'].sum() * 1000
            max_error_pct = df_balance['relative_error_pct'].max()
            
            print(f"⚖️  Energy balance: Losses={total_losses_kwh:.2f} kWh, Max error={max_error_pct:.4f}%")
            
        except Exception as e:
            print(f"⚠️ Energy balance check failed: {e}")
    

    def _load_results(self, results_path, n_steps):
        """Load simulation results from CSV files"""
        results_path = Path(results_path)
        
        try:
            # Load power flow results
            load_data = pd.read_csv(results_path / "res_load" / "p_mw.csv", 
                                   index_col=0, delimiter=';')
            sgen_data = pd.read_csv(results_path / "res_sgen" / "p_mw.csv", 
                                   index_col=0, delimiter=';')
            storage_data = pd.read_csv(results_path / "res_storage" / "p_mw.csv", 
                                      index_col=0, delimiter=';')
            grid_data = pd.read_csv(results_path / "res_ext_grid" / "p_mw.csv",
                                   index_col=0, delimiter=';')
            
            # Aggregate data (convert to kW)
            total_load = load_data.sum(axis=1) * 1000
            total_pv = sgen_data.sum(axis=1) * 1000
            total_battery = storage_data.sum(axis=1) * 1000
            grid_power = grid_data.iloc[:, 0] * 1000

            # Load battery SOC from res_storage/soc_percent.csv
            soc_file = results_path / 'res_storage' / 'soc_percent.csv'
            if soc_file.exists():
                soc_data = pd.read_csv(soc_file, index_col=0, delimiter=';')
                # Average SOC across all batteries (or take first battery if only one)
                if len(soc_data.columns) > 0:
                    battery_soc = soc_data.mean(axis=1).reset_index(drop=True)
                else:
                    battery_soc = pd.Series([50] * n_steps)
            else:
                battery_soc = pd.Series([50] * n_steps)

            return {
                'load': total_load,
                'pv_generation': total_pv,
                'battery_power': total_battery,
                'battery_soc': battery_soc,
                'grid_power': grid_power,
                'time_hours': range(n_steps)
            }
        
        except Exception as e:
            print(f"⚠️ Warning: Could not load all results: {e}")
            return {
                'load': pd.Series([0] * n_steps),
                'pv_generation': pd.Series([0] * n_steps),
                'battery_power': pd.Series([0] * n_steps),
                'battery_soc': pd.Series([50] * n_steps),
                'grid_power': pd.Series([0] * n_steps),
                'time_hours': range(n_steps)
            }
    
    def _print_summary(self, scenario_name, results):
        """Print simulation summary"""
        print(f"\n{'='*60}")
        print(f"📈 RESULTS SUMMARY: {scenario_name}")
        print(f"{'='*60}\n")
        
        print("⚡ ENERGY BALANCE:")
        print(f"  Total Load:        {results['load'].sum():.1f} kWh")
        print(f"  Total PV Gen:      {results['pv_generation'].sum():.1f} kWh")
        print(f"  Battery Discharge: {results['battery_power'][results['battery_power'] > 0].sum():.1f} kWh")
        print(f"  Battery Charge:    {abs(results['battery_power'][results['battery_power'] < 0].sum()):.1f} kWh")
        
        # Calculate grid import/export
        grid_import = results['grid_power'][results['grid_power'] > 0].sum()
        grid_export = abs(results['grid_power'][results['grid_power'] < 0].sum())
        
        print(f"  Grid Import:       {grid_import:.1f} kWh")
        print(f"  Grid Export:       {grid_export:.1f} kWh")

        try:
            balance_file = results['output_path'] / "energy_balance.csv"
            if balance_file.exists():
                df_balance = pd.read_csv(balance_file)
                total_losses = df_balance['total_losses_mw'].sum() * 1000
                
                print(f"  Total Losses:      {total_losses:.2f} kWh")
                
                # Loss percentage
                total_load = results['load'].sum()
                if total_load > 0:
                    loss_pct = (total_losses / total_load) * 100
                    print(f"  Loss Percentage:   {loss_pct:.2f}%")
        except:
            pass
        
        # Self-sufficiency
        if results['load'].sum() > 0:
            self_suff = min(100, (results['pv_generation'].sum() / results['load'].sum()) * 100)
            print(f"  Self-Sufficiency:  {self_suff:.1f}%")
        
        print(f"\n{'='*60}\n")
    
    def get_result(self, scenario_name):
        """Retrieve results for a specific scenario"""
        return self.results.get(scenario_name)
    
    def export_results(self, scenario_name, format='csv'):
        """Export results in various formats"""
        if scenario_name not in self.results:
            raise ValueError(f"No results for scenario '{scenario_name}'")
        
        result = self.results[scenario_name]
        output_path = result['output_path']
        
        if format == 'csv':
            print(f"📁 CSV results available at: {output_path}")
        
        elif format == 'excel':
            excel_path = output_path / 'summary.xlsx'
            with pd.ExcelWriter(excel_path) as writer:
                df_energy = pd.DataFrame(result['data'])
                df_energy.to_excel(writer, sheet_name='Energy_Flows')
            
            print(f"📊 Excel summary exported to: {excel_path}")
        
        return output_path