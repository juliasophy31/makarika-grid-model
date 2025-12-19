"""
Optimized Battery Controller

Applies pre-computed battery schedules from CSV files  (created by standalone_optimizer) to pandapower simulation.
This is a simple schedule playback controller.
"""

import numpy as np
import pandas as pd
from pandapower.control.basic_controller import Controller
from pathlib import Path


class OptimizedBatteryController(Controller):
    """
    Controller that applies pre-computed battery schedules from CSV files.
    
    Expects CSV format:
    timestep,battery1_power_kw,battery1_soc_percent,battery2_power_kw,battery2_soc_percent
    0,-2.5,55.0,-3.0,57.0
    1,-1.0,60.0,-1.5,62.0
    ...
    """
    
    def __init__(self, net, battery1_name, battery2_name, schedule_file, **kwargs):
        """
        Initialize optimized battery controller
        
        Parameters:
        -----------
        net : pandapower network
        battery1_name : str
            Name of battery 1 (e.g., "Battery_bus10")
        battery2_name : str
            Name of battery 2 (e.g., "Battery_bus18")
        schedule_file : str
            Path to CSV file with battery schedules
        """
        # Initialize controller with proper order
        super().__init__(net, in_service=True, order=1, level=0,
                        initial_run=True, recycle=True, **kwargs)
        
        self.net = net
        self.battery1_name = battery1_name
        self.battery2_name = battery2_name
        self.schedule_file = schedule_file
        self.timestep_h = 1.0
        
        # Get battery indices
        self.battery1_idx = net.storage[net.storage.name == battery1_name].index[0]
        self.battery2_idx = net.storage[net.storage.name == battery2_name].index[0]
        
        # Register both batteries as controlled elements
        self.controlled_elements = {
            'storage': [self.battery1_idx, self.battery2_idx]
        }
        self.element_index = [self.battery1_idx, self.battery2_idx]
        
        # Get battery parameters
        batt1_params = net.storage.loc[self.battery1_idx]
        batt2_params = net.storage.loc[self.battery2_idx]
        
        self.battery1_capacity_kwh = batt1_params['max_e_mwh'] * 1000
        self.battery1_efficiency = batt1_params['efficiency_percent'] / 100
        
        self.battery2_capacity_kwh = batt2_params['max_e_mwh'] * 1000
        self.battery2_efficiency = batt2_params['efficiency_percent'] / 100
        
        # Load schedules
        self.schedules = None
        self.has_run_this_step = False
        
        # Initialize SOC history tracking (required for simulator to copy SOC to results)
        self.battery1_soc_history = []
        self.battery2_soc_history = []
        
        print(f"\nOptimizedBatteryController initialized:")
        print(f"  Battery 1: {battery1_name} ({self.battery1_capacity_kwh:.1f} kWh)")
        print(f"  Battery 2: {battery2_name} ({self.battery2_capacity_kwh:.1f} kWh)")
        print(f"  Schedule file: {schedule_file}")
        
        # Load schedule file
        self._load_schedules()
        
        # Reset network state
        self._reset_network_state()
    
    def _load_schedules(self):
        """Load battery schedules from CSV/Excel file"""
        # Check if it's a relative path, make it relative to venv/data/optimization_results/
        if not Path(self.schedule_file).is_absolute():
            schedule_path = Path("venv/data/optimization_results") / self.schedule_file
        else:
            schedule_path = Path(self.schedule_file)
        
        if not schedule_path.exists():
            print(f"⚠️  Warning: Schedule file not found: {self.schedule_file}")
            print("    Creating default schedule (all zeros)")
            # Create default 24-hour schedule with zeros
            self.schedules = pd.DataFrame({
                'timestep': range(24),
                'battery1_power_kw': [0.0] * 24,
                'battery1_soc_percent': [50.0] * 24,
                'battery2_power_kw': [0.0] * 24,
                'battery2_soc_percent': [50.0] * 24
            })
            return
        
        try:
            # Read file based on extension
            if schedule_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(schedule_path)
                print(f"✅ Loaded Excel file: {len(df)} timesteps")
                
                # Map Excel columns to expected format
                column_mapping = {
                    'Hour': 'timestep',
                    'Bat10_Net_kW': 'battery1_power_kw', 
                    'SOC10_percent': 'battery1_soc_percent',
                    'Bat18_Net_kW': 'battery2_power_kw',
                    'SOC18_percent': 'battery2_soc_percent'
                }
                
                # Check if required source columns exist
                missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in Excel file: {missing_cols}")
                
                # Extract and rename columns
                self.schedules = df[list(column_mapping.keys())].rename(columns=column_mapping)
                
            else:
                # Assume CSV format with expected columns
                self.schedules = pd.read_csv(schedule_path)
                print(f"✅ Loaded CSV file: {len(self.schedules)} timesteps")
                
                # Verify required columns for CSV
                required_cols = ['battery1_power_kw', 'battery1_soc_percent', 
                               'battery2_power_kw', 'battery2_soc_percent']
                missing_cols = [col for col in required_cols if col not in self.schedules.columns]
                
                if missing_cols:
                    raise ValueError(f"Missing columns in CSV file: {missing_cols}")
            
            print(f"   Battery 1 power range: {self.schedules['battery1_power_kw'].min():.1f} to {self.schedules['battery1_power_kw'].max():.1f} kW")
            print(f"   Battery 2 power range: {self.schedules['battery2_power_kw'].min():.1f} to {self.schedules['battery2_power_kw'].max():.1f} kW")
            
        except Exception as e:
            print(f"❌ Error loading schedule file: {e}")
            print("    Creating default schedule (all zeros)")
            # Create default schedule as fallback
            self.schedules = pd.DataFrame({
                'timestep': range(24),
                'battery1_power_kw': [0.0] * 24,
                'battery1_soc_percent': [50.0] * 24,
                'battery2_power_kw': [0.0] * 24,
                'battery2_soc_percent': [50.0] * 24
            })
    
    def _reset_network_state(self):
        """Reset network to clean state after initialization"""
        # Reset both batteries to initial state
        self.net.storage.at[self.battery1_idx, 'p_mw'] = 0.0
        self.net.storage.at[self.battery1_idx, 'soc_percent'] = 50.0
        
        self.net.storage.at[self.battery2_idx, 'p_mw'] = 0.0
        self.net.storage.at[self.battery2_idx, 'soc_percent'] = 50.0
    
    def control_step(self, net):
        """Apply scheduled power and SOC at current timestep"""
        try:
            self.has_run_this_step = True
            
            t = net.time_step
            print(f"🔋 OptimizedBatteryController control_step t={t}")
            
            if self.schedules is None or t >= len(self.schedules):
                print(f"   No schedule available for timestep {t}")
                return
            
            # Get scheduled values
            battery1_power_kw = self.schedules.loc[t, 'battery1_power_kw']
            battery1_soc = self.schedules.loc[t, 'battery1_soc_percent']
            battery2_power_kw = self.schedules.loc[t, 'battery2_power_kw']
            battery2_soc = self.schedules.loc[t, 'battery2_soc_percent']
            
            # Apply Battery 1 schedule
            battery1_power_mw = battery1_power_kw / 1000  # Convert kW to MW
            net.storage.at[self.battery1_idx, 'p_mw'] = battery1_power_mw
            net.storage.at[self.battery1_idx, 'soc_percent'] = battery1_soc
            
            # Apply Battery 2 schedule  
            battery2_power_mw = battery2_power_kw / 1000  # Convert kW to MW
            net.storage.at[self.battery2_idx, 'p_mw'] = battery2_power_mw
            net.storage.at[self.battery2_idx, 'soc_percent'] = battery2_soc
            
            # Store SOC values in history for simulator to copy to results
            self.battery1_soc_history.append(battery1_soc)
            self.battery2_soc_history.append(battery2_soc)
            
            print(f"   {self.battery1_name}: {battery1_power_kw:+5.1f}kW, SOC: {battery1_soc:.1f}%")
            print(f"   {self.battery2_name}: {battery2_power_kw:+5.1f}kW, SOC: {battery2_soc:.1f}%")
            
        except Exception as e:
            print(f"❌ OptimizedBatteryController control_step error: {e}")
            # Set batteries to safe state
            net.storage.at[self.battery1_idx, 'p_mw'] = 0.0
            net.storage.at[self.battery2_idx, 'p_mw'] = 0.0
    
    def is_converged(self, net):
        """Controller convergence check"""
        if not self.has_run_this_step:
            return False
        return True
    
    def finalize_step(self, net, time_step):
        """Called after each timestep"""
        self.has_run_this_step = False
    
    def time_step(self, net, time):
        """Called to set the current time step"""
        self.net.time_step = time
    
    def get_battery_power(self, timestep):
        """Get battery power at specific timestep (for external access)"""
        if self.schedules is None or timestep >= len(self.schedules):
            return {
                'battery1_power_kw': 0.0,
                'battery2_power_kw': 0.0,
                'total_battery_power_kw': 0.0
            }
        
        battery1_power = self.schedules.loc[timestep, 'battery1_power_kw']
        battery2_power = self.schedules.loc[timestep, 'battery2_power_kw']
        
        return {
            'battery1_power_kw': battery1_power,
            'battery2_power_kw': battery2_power,
            'total_battery_power_kw': battery1_power + battery2_power
        }
    
    def get_schedule_summary(self):
        """Get summary of loaded schedule"""
        if self.schedules is None:
            return "No schedule loaded"
        
        return {
            'timesteps': len(self.schedules),
            'battery1_total_energy': self.schedules['battery1_power_kw'].sum(),
            'battery2_total_energy': self.schedules['battery2_power_kw'].sum(),
            'battery1_cycles': abs(self.schedules['battery1_power_kw']).sum() / self.battery1_capacity_kwh,
            'battery2_cycles': abs(self.schedules['battery2_power_kw']).sum() / self.battery2_capacity_kwh
        }