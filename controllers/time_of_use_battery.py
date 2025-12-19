# Time-of-Use Battery Controller
"""
Time-of-Use (TOU) Battery Controller
=====================================

This controller implements a price-responsive battery control strategy that 
charges during low-cost periods and discharges during high-cost periods to 
minimize electricity costs under time-of-use tariff structures.

Control Strategy:
-----------------
The controller uses a two-period pricing structure (peak/off-peak):

OFF-PEAK HOURS (low electricity prices):
  - If SOC < 100%: Charge from grid + PV to prepare for peak hours
  - If SOC > 90% AND local loads need power: Discharge to meet deficit
  
PEAK HOURS (high electricity prices):
  - Discharge to meet local region loads
  - Avoid expensive grid imports by using stored energy
  - Only export if all local loads are satisfied first

How to use:
-----------
This controller is typically created by the ControllerFactory and integrated
into pandapower time-series simulations. It is NOT meant to be instantiated
directly by users.


Control Logic Details:
----------------------
At each timestep, the controller:
1. Reads current time and determines if peak or off-peak period
2. Calculates local region load from all connected loads
3. Calculates available PV generation
4. Computes net deficit/surplus (load - PV generation)
5. Makes control decision:
6. Applies power limits and SOC constraints
7. Updates battery setpoint in pandapower network

Battery Constraints:
--------------------
- Maximum charge/discharge power: ±5 kW (configurable)
- SOC limits: 10% to 100%
- Round-trip efficiency: Applied in pandapower storage model
- Capacity: 20 kWh (configurable in network definition)

Dependencies:
-------------
- pandapower.control.basic_controller.Controller
- NumPy for array operations
"""

# controllers/tou_battery.py
# Time-of-Use Battery Controller with system-wide optimization

from pandapower.control.basic_controller import Controller

class TOUBattery(Controller):
    """
    Time-of-Use Battery Controller - Price-Responsive Control
    
    Key Features:
    - Local region load awareness (loads behind same transformer)
    - Smart export: only export when local loads satisfied
    - Off-peak charging: charge during cheap hours if not full
    - Off-peak discharge: discharge if SOC > 90% and deficit exists (avoid waste)
    - Two-period pricing: peak vs off-peak (no shoulder)

    Control Logic:
    1. OFF-PEAK:
       - If SOC > 90% and deficit exists: Discharge to meet load
       - If SOC < 100%: Charge from cheap grid + PV to prepare for peak
    2. PEAK: Discharge to meet local region load, avoid expensive import
    3. EXPORT: Only if all local loads satisfied first
    """
    
    def __init__(self, net, pv_name, battery_name, load_names=None, bus_name=None,
                 timestep_h=1.0, market_config=None, **kwargs):
        super().__init__(net, in_service=True, order=1, level=0,
                        initial_run=True, recycle=True, **kwargs)
        self.net = net
        self.timestep_h = timestep_h
        self.bus_name = bus_name

        # Identify battery and PV indices
        self.pv_idx = net.sgen[net.sgen.name == pv_name].index[0]
        self.batt_idx = net.storage[net.storage.name == battery_name].index[0]
        self.storage_idx = self.batt_idx

        # Identify load indices for local region
        if load_names:
            self.load_idxs = [net.load[net.load.name == n].index[0] for n in load_names]
        else:
            # Fallback: use all loads (old behavior for backward compatibility)
            self.load_idxs = net.load.index.tolist()

        # Define controlled elements
        self.controlled_elements = {'storage': [self.batt_idx]}
        self.element_index = [self.batt_idx]
        
        # Market configuration (from defaults)
        if market_config is None:
            market_config = {}

        self.peak_hours = market_config.get('peak_hours', [7, 8, 9, 10, 17, 18, 19, 20])
        self.peak_price = market_config.get('peak_import_price', 0.35)
        self.offpeak_price = market_config.get('offpeak_import_price', 0.23)
        self.export_price = market_config.get('export_price', 0.13)

        # NEW: Charging strategy parameters
        self.offpeak_import_charge_rate = market_config.get('offpeak_import_charge_rate', 0.8)  # 0.0 = PV-only, 1.0 = full import, used to be 1.0
        self.offpeak_max_soc = market_config.get('offpeak_max_soc', 100)  # Target SOC during off-peak
        self.offpeak_discharge_threshold = market_config.get('offpeak_discharge_threshold', 50)  # SOC threshold to allow off-peak discharge, usec to be 90
        self.peak_discharge_min_soc = market_config.get('peak_discharge_min_soc', 10)  # Minimum SOC during peak discharge
        self.peak_export_min_soc = market_config.get('peak_export_min_soc', 10)  # Reserve during peak export

        self.has_run_this_step = False
        self.soc_history = []
        self.operation_log = []

        bus_info = f" at {bus_name}" if bus_name else ""
    

    def control_step(self, net):
        bus_info = f"[{self.bus_name}] " if self.bus_name else ""
        
        self.has_run_this_step = True
        
        # Get current state
        timestep = getattr(net, 'time_step', 0)
        current_hour = timestep % 24
        is_peak = current_hour in self.peak_hours
        
        # Get LOCAL power flows (only this region behind the same transformer)
        local_pv = net.sgen.at[self.pv_idx, 'p_mw']  # Only this battery's PV
        local_load = net.load.loc[self.load_idxs, 'p_mw'].sum()  # Only loads behind same trafo
        soc = net.storage.at[self.batt_idx, 'soc_percent']

        period_name = "PEAK" if is_peak else "OFF-PEAK"

    

        # Battery parameters
        max_p = net.storage.at[self.batt_idx, 'max_p_mw']
        min_p = net.storage.at[self.batt_idx, 'min_p_mw']
        max_e = net.storage.at[self.batt_idx, 'max_e_mwh']
        eff = net.storage.at[self.batt_idx, 'efficiency_percent'] / 100.0

        # Initialize decision variables
        p_batt = 0.0
        soc_new = soc
        decision = "IDLE"

        # Calculate local net load (positive = deficit, negative = surplus)
        local_net_load = local_load - local_pv

        # REFINED TOU CONTROL LOGIC
        if not is_peak:
            # ===== OFF-PEAK PERIOD: CHARGE BATTERY OR DISCHARGE IF HIGHLY CHARGED =====
            # Priority: Charge battery to prepare for peak period
            # Exception: If SOC > threshold and local deficit exists, discharge to avoid waste

            if local_net_load > 0 and soc > self.offpeak_discharge_threshold:
                # Local deficit exists and battery is highly charged (>threshold)
                # Discharge to help meet load rather than import from grid
                deficit_power = local_net_load
                discharge_power = min(max_p, deficit_power)
                energy_needed = discharge_power * self.timestep_h / eff
                soc_decrease = (energy_needed / max_e) * 100

                if soc - soc_decrease >= self.peak_discharge_min_soc:
                    p_batt = -discharge_power  # Negative = discharging (pandapower convention)
                    soc_new = soc - soc_decrease
                    decision = "OFFPEAK_DISCHARGE_HIGH_SOC"
                else:
                    # Partial discharge to maintain minimum SOC
                    available_energy = (soc - self.peak_discharge_min_soc) / 100 * max_e
                    actual_discharge_power = available_energy * eff / self.timestep_h
                    p_batt = -min(discharge_power, actual_discharge_power)  # Negative = discharging
                    soc_new = max(self.peak_discharge_min_soc, soc - soc_decrease)
                    decision = "OFFPEAK_DISCHARGE_LIMITED"

            elif soc < self.offpeak_max_soc:
                if local_net_load <= 0:
                    # Local region has PV surplus - use it to charge
                    surplus_power = abs(local_net_load)
                    charge_power = min(max_p, surplus_power)
                    energy_to_store = charge_power * self.timestep_h * eff
                    soc_increase = (energy_to_store / max_e) * 100

                    if soc + soc_increase <= self.offpeak_max_soc:
                        p_batt = charge_power  # Positive = charging (pandapower convention)
                        soc_new = soc + soc_increase
                        decision = "OFFPEAK_CHARGE_PV"
                    else:
                        remaining_capacity = (self.offpeak_max_soc - soc) / 100 * max_e
                        actual_charge_power = remaining_capacity / (self.timestep_h * eff)
                        p_batt = min(charge_power, actual_charge_power)  # Positive = charging
                        soc_new = self.offpeak_max_soc
                        decision = "OFFPEAK_CHARGE_PV_FULL"
                else:
                    # Local region has deficit - charge from cheap grid anyway
                    # This is the key behavior: buy cheap power to store for peak
                    charge_power = self.offpeak_import_charge_rate * max_p  # Configurable charge rate
                    energy_to_store = charge_power * self.timestep_h * eff
                    soc_increase = (energy_to_store / max_e) * 100

                    if soc + soc_increase <= self.offpeak_max_soc:
                        p_batt = charge_power  # Positive = charging (pandapower convention)
                        soc_new = soc + soc_increase
                        decision = "OFFPEAK_CHARGE_GRID"
                    else:
                        remaining_capacity = (self.offpeak_max_soc - soc) / 100 * max_e
                        actual_charge_power = remaining_capacity / (self.timestep_h * eff)
                        p_batt = min(charge_power, actual_charge_power)  # Positive = charging
                        soc_new = self.offpeak_max_soc
                        decision = "OFFPEAK_CHARGE_FULL"
            else:
                decision = "OFFPEAK_BATTERY_FULL"
        
        else:
            # ===== PEAK PERIOD: DISCHARGE TO MEET LOAD =====

            if soc > 10:  # Only discharge if sufficient SOC
                if local_net_load > 0:
                    # Local region has deficit - discharge to reduce expensive grid import
                    deficit_power = local_net_load
                    discharge_power = min(max_p, deficit_power)
                    energy_needed = discharge_power * self.timestep_h / eff
                    soc_decrease = (energy_needed / max_e) * 100
                    
                    if soc - soc_decrease >= 10:
                        p_batt = -discharge_power  # Negative = discharging (pandapower convention)
                        soc_new = soc - soc_decrease
                        decision = "PEAK_DISCHARGE"
                    else:
                        # Can only partially discharge
                        available_energy = (soc - 10) / 100 * max_e
                        actual_discharge_power = available_energy * eff / self.timestep_h
                        p_batt = -min(discharge_power, actual_discharge_power)  # Negative = discharging
                        soc_new = 10.0
                        decision = "PEAK_DISCHARGE_LIMITED"
                
                else:
                    # Local region has PV surplus - all loads are satisfied
                    # Discharge to export surplus, but maintain minimum SOC reserve

                    if soc > self.peak_export_min_soc:
                        surplus_power = abs(local_net_load)
                        discharge_power = min(max_p, surplus_power)  # Use full power
                        energy_needed = discharge_power * self.timestep_h / eff
                        soc_decrease = (energy_needed / max_e) * 100

                        if soc - soc_decrease >= 10:
                            # Can discharge fully without hitting minimum
                            p_batt = -discharge_power  # Negative = discharging (pandapower convention)
                            soc_new = soc - soc_decrease
                            decision = "PEAK_EXPORT_SURPLUS"
                        else:
                            # Partial discharge to maintain 10% minimum
                            available_energy = (soc - 10) / 100 * max_e
                            actual_discharge_power = available_energy * eff / self.timestep_h
                            p_batt = -actual_discharge_power  # Negative = discharging
                            soc_new = 10.0
                            decision = "PEAK_EXPORT_LIMITED"
                    else:
                        decision = "PEAK_SOC_TOO_LOW"
            else:
                decision = "PEAK_SOC_TOO_LOW"
    
        # Record SOC at START of this timestep (before actions are applied)
        # This matches the optimizer output format: SOC at start of each hour
        self.soc_history.append(soc)
        
        # Clamp battery power and SOC
        p_batt = max(min_p, min(max_p, p_batt))
        soc_new = max(0.0, min(100.0, soc_new))

        # Apply battery power and SOC
        net.storage.at[self.batt_idx, 'p_mw'] = p_batt
        net.storage.at[self.batt_idx, 'soc_percent'] = soc_new

        # Calculate actual grid power after battery action
        # Grid power = Load - PV - Battery discharge (or + Battery charge)
        grid_power = local_load - local_pv - p_batt
        
        # Determine price based on period and direction
        if grid_power > 0:  # Importing
            price = self.peak_price if is_peak else self.offpeak_price
        else:  # Exporting
            price = self.export_price
        
        # Log operation for economic analysis
        self.operation_log.append({
            'timestep': timestep,
            'hour': current_hour,
            'period': period_name,
            'decision': decision,
            'local_pv_mw': local_pv,
            'local_load_mw': local_load,
            'battery_mw': p_batt,
            'grid_mw': grid_power,
            'soc_percent': soc_new,
            'price_kwh': abs(price),
            'grid_cost': grid_power * price * self.timestep_h  # Can be negative (revenue)
        })


    def is_converged(self, net):
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
    
    
    def save_operation_log(self, filename=None):
        """Save detailed operation log for economic analysis"""
        import pandas as pd
        import os
        
        if filename is None:
            bus_suffix = f"_{self.bus_name}" if self.bus_name else ""
            filename = f'results/tou_operation_log{bus_suffix}.csv'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        log_df = pd.DataFrame(self.operation_log)
        log_df.to_csv(filename, index=False)
        
        
        
        return log_df