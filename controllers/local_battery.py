# local_battery.py
"""
Local Battery Controller 
=======================================================

Simple, autonomous battery control that responds to local supply/demand 
balance without requiring communication or forecasting.

What it does:
-------------
- Charges when local PV generation exceeds local loads
- Discharges when local loads exceed local PV generation
- Operates independently based only on real-time measurements
- No coordination with other batteries or grid pricing

Control Logic:
--------------
IF PV > Load (surplus):
  → Charge battery (if SOC < 100%)
  → Store excess energy locally
  
IF Load > PV (deficit):
  → Discharge battery (if SOC > 10%)
  → Supply local demand from stored energy)

How to use:
-----------
Created via ControllerFactory, not directly instantiated:
"""

from pandapower.control.basic_controller import Controller

class SimpleBattery(Controller):
    
    def __init__(self, net, pv_name, battery_name, load_names, bus_name=None, timestep_h=1.0, **kwargs):
        super().__init__(net, in_service=True, order=1, level=0, 
                        initial_run=True, recycle=True, **kwargs)
        self.net = net
        self.timestep_h = timestep_h
        self.bus_name = bus_name

        # Identify indices
        self.pv_idx = net.sgen[net.sgen.name == pv_name].index[0]
        self.batt_idx = net.storage[net.storage.name == battery_name].index[0]
        self.storage_idx = self.batt_idx  # For use in simulator's SOC saving
        self.load_idxs = [net.load[net.load.name == n].index[0] for n in load_names]

        # Define controlled elements
        self.controlled_elements = {'storage': [self.batt_idx]}
        self.element_index = [self.batt_idx]
        
        self.has_run_this_step = False
        self.soc_history = []
        
        bus_info = f" at {bus_name}" if bus_name else ""
       

    def control_step(self, net):
        bus_info = f"[{self.bus_name}] " if self.bus_name else ""
        
        self.has_run_this_step = True
        
        pv_p = net.sgen.at[self.pv_idx, 'p_mw']
        load_p = net.load.loc[self.load_idxs, 'p_mw'].sum()
        soc = net.storage.at[self.batt_idx, 'soc_percent']
        
        timestep = getattr(net, 'time_step', 0)

        max_p = net.storage.at[self.batt_idx, 'max_p_mw']
        min_p = net.storage.at[self.batt_idx, 'min_p_mw']
        max_e = net.storage.at[self.batt_idx, 'max_e_mwh']
        eff = net.storage.at[self.batt_idx, 'efficiency_percent'] / 100.0

        p_batt = 0.0
        soc_new = soc
        
        if pv_p > load_p:  # PV surplus - try to charge
            if soc < 100:
                surplus_power = pv_p - load_p
                charge_power = min(max_p, surplus_power)
                energy_to_store = charge_power * self.timestep_h * eff
                soc_increase = (energy_to_store / max_e) * 100
                
                if soc + soc_increase <= 100:
                    p_batt = charge_power  # Positive = charging (pandapower convention)
                    soc_new = soc + soc_increase
                else:
                    remaining_capacity = (100 - soc) / 100 * max_e
                    actual_charge_power = remaining_capacity / (self.timestep_h * eff)
                    p_batt = min(charge_power, actual_charge_power)  # Positive = charging
                    soc_new = 100.0
            else:
                p_batt = 0
                soc_new = soc
                
        else:  # PV deficit - try to discharge
            if soc > 10:
                deficit_power = load_p - pv_p
                discharge_power = min(max_p, deficit_power)
                energy_needed = discharge_power * self.timestep_h / eff
                soc_decrease = (energy_needed / max_e) * 100
                
                if soc - soc_decrease >= 10:
                    p_batt = -discharge_power  # Negative = discharging (pandapower convention)
                    soc_new = soc - soc_decrease
                else:
                    available_energy = (soc - 10) / 100 * max_e
                    actual_discharge_power = available_energy * eff / self.timestep_h
                    p_batt = -min(discharge_power, actual_discharge_power)  # Negative = discharging
                    soc_new = 10.0
            else:
                p_batt = 0
                soc_new = soc
    
        # Record SOC at START of this timestep (before actions are applied)
        # This matches the optimizer output format: SOC at start of each hour
        self.soc_history.append(soc)
        
        # Clamp battery power and SOC
        p_batt = max(min_p, min(max_p, p_batt))
        soc_new = max(0.0, min(100.0, soc_new))

        # Apply battery power and SOC
        net.storage.at[self.batt_idx, 'p_mw'] = p_batt
        net.storage.at[self.batt_idx, 'soc_percent'] = soc_new
        
        
    def is_converged(self, net):
        bus_info = f"[{self.bus_name}] " if self.bus_name else ""
        if not self.has_run_this_step:
            return False
        return True
    
    def finalize_step(self, net, time_step):
        """Called after each time step"""
        self.has_run_this_step = False
    
    def time_step(self, net, time):
        """Called to set the current time step"""
        self.net.time_step = time
    
    def save_soc_history(self, filename=None):
        """Save SOC history to CSV file"""
        import pandas as pd
        import os
        
        if filename is None:
            bus_suffix = f"_{self.bus_name}" if self.bus_name else ""
            filename = f'results/battery_soc{bus_suffix}.csv'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        soc_df = pd.DataFrame({
            'timestep': range(len(self.soc_history)),
            'soc_percent': self.soc_history,
            'battery_location': [self.bus_name] * len(self.soc_history) if self.bus_name else ['unknown'] * len(self.soc_history)
        })
        
        soc_df.to_csv(filename, index=False)
        print(f"💾 SOC history saved to: {filename}")