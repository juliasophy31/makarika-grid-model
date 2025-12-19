"""
================================================================================
USER GUIDE - STANDALONE BATTERY OPTIMIZER
================================================================================

What it does:
-------------
This standalone optimizer finds the optimal battery charging and discharging 
schedules for a two-battery microgrid system to minimize electricity costs over
a 24-hour period. It uses convex optimization (CVXPY) to solve the problem 
described in the mathematical formulation above.

The optimizer considers:
- Time-varying electricity prices (peak vs off-peak)
- PV solar generation profiles
- Load demand patterns (with or without marae events)
- Battery capacity and power limits
- Round-trip efficiency losses
- Export revenue from excess generation

Key Features:
-------------
1. Two independent battery systems (Bus 10 and Bus 18)
2. Aggregated load zones per battery
3. Time-of-use tariff optimization
4. PV generation integration
5. State-of-charge tracking with cycling constraints
6. Power balance verification
7. Excel export for detailed analysis

How to use:
-----------

OPTION 1: Generate all scenarios at once
    python "standalone optimizer.py"
    
    This runs the generate_all_scenarios() function which creates optimization
    results for all four combinations:
    - summer_marae
    - summer_no_marae  
    - winter_marae
    - winter_no_marae

OPTION 2: Run a single custom scenario

    1. Edit the main section at the bottom:
       
       loader = DataLoader()
       loader.load_profiles(season='summer', marae=False, C_pv10=18, C_pv18=18)
       scenario_name = 'summer_no_marae'
    
    2. Run the script:
       python "standalone optimizer.py"

Output Files:
-------------
Results are saved to: venv/data/optimization_results/


File naming: {scenario_name}_optimization_results.xlsx

Parameters You Can Modify:
---------------------------
In DataLoader.load_profiles():
- season: 'summer' or 'winter'
- marae: True (with event) or False (normal operation)
- C_pv10: PV capacity at Bus 10 in kW (default: 18 kW)
- C_pv18: PV capacity at Bus 18 in kW (default: 18 kW)


Dependencies:
-------------
- numpy: Array operations
- pandas: Data loading and export
- cvxpy: Convex optimization framework
- ecos: Solver backend (installed with cvxpy)
- openpyxl: Excel file handling

Input Data Requirements:
------------------------
The script expects the following files in venv/data/:

1. loads_profile.xlsx

2. pv_standard_profile.csv

Notes:
------
- The optimizer assumes perfect foresight (knows future loads and generation)
- This is the theoretical minimum cost baseline for comparison
- In practice, forecasting errors will increase actual costs
- The model uses hourly timesteps (Δt = 1 hour)
- Battery cycling constraint ensures SOC returns to initial level by end of day

================================================================================
MATHEMATICAL PROBLEM STATEMENT - BATTERY OPTIMIZATION
================================================================================
OBJECTIVE:
Minimize total electricity cost over time horizon T

DECISION VARIABLES 

    Global:
    - P_import[t]    : Grid import power at time t (kW)
    - P_export[t]    : Grid export power at time t (kW)
    - P_grid10[t]    : Net power flow from grid to zone 10 at time t (kW)
    - P_grid18[t]    : Net power flow from grid to zone 18 at time t (kW)

    Zone bus_10:
    - P_bat10_dis[t] : Battery discharging power at time t (kW) 
    - P_bat10_chg[t] : Battery charging power at time t (kW)
    - SOC10[t]       : State of charge at time t (%)

    zone bus_18: 
    - P_bat18_dis[t] : Battery discharging power at time t (kW) 
    - P_bat18_chg[t]  : Battery charging power at time t (kW)
    - SOC18[t]       : State of charge at time t (%)


PARAMETERS:

    Global:
    - T           : Time horizon (hours)
    - η_bat       : Battery round-trip efficiency (0.9)
    - C_bat       : Battery capacity (20 kWh)
    - p_bat_max   : Maximum battery discharge power (5.0 kW)
    - p_bat_min   : Maximum battery charge power (-5.0 kW)
    - SOC_max     : Maximum state of charge (100%)
    - SOC_min     : Minimum state of charge (10%)
    - SOC_0       : Initial state of charge (50%)
    - cf_pv [t]   : PV capacity factor at time t
    - c_peak      : Peak import price (0.33 $/kWh)
    - c_offpeak   : Off-peak import price (0.23 $/kWh)
    - c_export    : Export revenue (0.13 $/kWh)
    - T_peak      : Peak hours [7, 8, 9, 10, 17, 18, 19, 20]

    Zone specific:
    - p_load_zone10 [t]   : aggregated load at time t in zone10 (kW)
    - p_load_zone18[t]    : aggregated load at time t in zone18 (kW)
    - C_pv10              : PV capacity  (kWh)
    - C_pv18              : PV capacity at bus 18 (kWh)

OBJECTIVE FUNCTION:
Minimize: Σ[t∈T_peak] P_import[t] * c_peak + Σ[t∉T_peak] P_import[t] * c_offpeak - Σ[t∈T] P_export[t] * c_export

CONSTRAINTS:

1. Zone 10 Power Balance:
   P_grid10[t] + C_pv10 * cf_pv[t] + P_bat10_dis[t] - P_bat10_chg[t] = p_load_zone10[t]  ∀t ∈ T

2. Zone 18 Power Balance:
   P_grid18[t] + C_pv18 * cf_pv[t] + P_bat18_dis[t] - P_bat18_chg[t] = p_load_zone18[t]  ∀t ∈ T

3. Grid Connection Constraint:
   P_import[t] - P_export[t] = P_grid10[t] + P_grid18[t]  ∀t ∈ T

4. Individual Battery Power Limits:
   0 ≤ P_bat10_dis[t] ≤ p_bat_max ∀t ∈ T
   0 ≤ P_bat10_chg[t] ≤ p_bat_max ∀t ∈ T
   0 ≤ P_bat18_dis[t] ≤ p_bat_max ∀t ∈ T
   0 ≤ P_bat18_chg[t] ≤ p_bat_max ∀t ∈ T

5. Individual State of Charge Limits:
   SOC_min ≤ SOC10[t] ≤ SOC_max  ∀t ∈ T
   SOC_min ≤ SOC18[t] ≤ SOC_max  ∀t ∈ T

6. Individual State of Charge Dynamics:
   SOC10[t+1] = SOC10[t] + ((P_bat10_chg[t] * η_bat - P_bat10_dis[t]) * Δt / C_bat) * 100  ∀t ∈ T
   SOC18[t+1] = SOC18[t] + ((P_bat18_chg[t] * η_bat - P_bat18_dis[t]) * Δt / C_bat) * 100  ∀t ∈ T

7. Initial Conditions:
   SOC10[0] = SOC_0
   SOC18[0] = SOC_0

8. Non-negativity:
   P_import[t] ≥ 0, P_export[t] ≥ 0  ∀t ∈ T
   P_bat10_dis[t] ≥ 0, P_bat10_chg[t] ≥ 0  ∀t ∈ T
   P_bat18_dis[t] ≥ 0, P_bat18_chg[t] ≥ 0  ∀t ∈ T

9. Grid Power Flow (can be positive or negative):
   P_grid10[t] ∈ ℝ, P_grid18[t] ∈ ℝ  ∀t ∈ T

10. Mutual Exclusivity:
    P_import[t] * P_export[t] = 0  ∀t ∈ T
    P_bat10_dis[t] * P_bat10_chg[t] = 0  ∀t ∈ T
    P_bat18_dis[t] * P_bat18_chg[t] = 0  ∀t ∈ T

11. final soc:
    SOC10[23] ≥ SOC_0
    SOC18[23] ≥ SOC_0


"""


import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir='venv/data'):
        self.data_dir = Path(data_dir)
    
    def load_profiles(self, season='summer', marae=False, C_pv10=18, C_pv18=18):
        """
        Load and process profiles for optimization
        
        Args:
            season (str): 'summer' or 'winter'
            marae (bool): True for with marae, False for no marae
            C_pv10 (float): PV capacity at bus 10 in kW
            C_pv18 (float): PV capacity at bus 18 in kW
        """
        # Determine sheet name based on season and marae activity
        marae_str = "marae" if marae else "no_marae"
        sheet_name = f"{season}_{marae_str}"
        
        # Load load profiles
        self.loads_df = pd.read_excel(self.data_dir / 'loads_profile.xlsx', 
                                     sheet_name=sheet_name, index_col=0)
        
        # Aggregate loads per battery zone (matching mathematical notation)
        self.p_load_zone10 = (self.loads_df['load_bus4_kw'].values + 
                              self.loads_df['load_bus10_kw'].values + 
                              self.loads_df['load_bus11_kw'].values)
        
        self.p_load_zone18 = (self.loads_df['load_bus17_kw'].values + 
                              self.loads_df['load_bus18_kw'].values + 
                              self.loads_df['load_bus20_kw'].values + 
                              self.loads_df['load_bus21_kw'].values)
        
        # Load PV capacity factors
        self.pv_df = pd.read_csv(self.data_dir / 'pv_standard_profile.csv',
                                delimiter=';', index_col=0)
        
        # Select capacity factor column based on season
        cf_column = f'capacity_factor_{season}'
        self.cf_pv = self.pv_df[cf_column].values
        
        # Store PV capacities (matching mathematical notation)
        self.C_pv10 = C_pv10
        self.C_pv18 = C_pv18
        
        # Calculate total PV generation per zone
        self.pv_gen_zone10 = self.C_pv10 * self.cf_pv
        self.pv_gen_zone18 = self.C_pv18 * self.cf_pv
        self.total_pv_generation = self.pv_gen_zone10 + self.pv_gen_zone18
        
        # Time horizon
        self.T = len(self.cf_pv)
        
        print(f"Loaded {season} profiles with {'marae' if marae else 'no marae'}")
        print(f"Time horizon: {self.T} hours")
        print(f"PV capacities: Zone10={C_pv10}kW, Zone18={C_pv18}kW")

def optimize_two_battery_system(data_loader):
    """
    Optimize two-battery system based on mathematical formulation
    
    Args:
        data_loader: DataLoader instance with loaded profiles
    """
    # PARAMETERS 
    η_bat = 0.9         # Battery round-trip efficiency
    C_bat = 20.0        # Battery capacity (kWh) 
    p_bat_max = 5.0     # Maximum battery power (kW)
    SOC_max = 100.0     # Maximum state of charge (%)
    SOC_min = 10.0      # Minimum state of charge (%)
    SOC_0 = 50.0        # Initial state of charge (%)
    c_peak = 0.33       # Peak import price ($/kWh)
    c_offpeak = 0.23    # Off-peak import price ($/kWh)
    c_export = 0.13     # Export price ($/kWh)
    T_peak = [7, 8, 9, 10, 17, 18, 19, 20]  # Peak hours
    
    T = data_loader.T   # Time horizon
    Δt = 1.0            # Time step (hour)
    
    # DECISION VARIABLES 
    # Global variables
    P_import = cp.Variable(T, nonneg=True)
    P_export = cp.Variable(T, nonneg=True) 
    P_net_zone10 = cp.Variable(T)  # Net power needed by zone 10 from grid
    P_net_zone18 = cp.Variable(T)  # Net power needed by zone 18 from grid
    
    # Zone 10 battery variables
    P_bat10_dis = cp.Variable(T, nonneg=True)
    P_bat10_chg = cp.Variable(T, nonneg=True)
    SOC10 = cp.Variable(T+1)
    
    # Zone 18 battery variables  
    P_bat18_dis = cp.Variable(T, nonneg=True)
    P_bat18_chg = cp.Variable(T, nonneg=True)
    SOC18 = cp.Variable(T+1)
    
    # OBJECTIVE FUNCTION
    cost = 0
    for t in range(T):
        if t in T_peak:
            cost += P_import[t] * c_peak
        else:
            cost += P_import[t] * c_offpeak
        cost -= P_export[t] * c_export
    
    objective = cp.Minimize(cost)
    
    # CONSTRAINTS
    constraints = []
    
    for t in range(T):
        # 1. Zone 10 Power Balance
        constraints.append(
            P_net_zone10[t] + data_loader.C_pv10 * data_loader.cf_pv[t] + 
            P_bat10_dis[t] - P_bat10_chg[t] == data_loader.p_load_zone10[t]
        )
        
        # 2. Zone 18 Power Balance  
        constraints.append(
            P_net_zone18[t] + data_loader.C_pv18 * data_loader.cf_pv[t] + 
            P_bat18_dis[t] - P_bat18_chg[t] == data_loader.p_load_zone18[t]
        )
        
        # 3. Grid Connection Constraint
        constraints.append(
            P_import[t] - P_export[t] == P_net_zone10[t] + P_net_zone18[t]
        )
        
        # 4. Battery Power Limits
        constraints.append(P_bat10_dis[t] <= p_bat_max)
        constraints.append(P_bat10_chg[t] <= p_bat_max)
        constraints.append(P_bat18_dis[t] <= p_bat_max)
        constraints.append(P_bat18_chg[t] <= p_bat_max)
        
        # 6. SOC Dynamics
        if t == 0:
            constraints.append(SOC10[0] == SOC_0)
            constraints.append(SOC18[0] == SOC_0)
        
        constraints.append(
            SOC10[t+1] == SOC10[t] + ((P_bat10_chg[t] * η_bat - P_bat10_dis[t]) * Δt / C_bat) * 100
        )
        constraints.append(
            SOC18[t+1] == SOC18[t] + ((P_bat18_chg[t] * η_bat - P_bat18_dis[t]) * Δt / C_bat) * 100
        )
        
        # 5. SOC Limits
        constraints.append(SOC10[t] >= SOC_min)
        constraints.append(SOC10[t] <= SOC_max)
        constraints.append(SOC18[t] >= SOC_min) 
        constraints.append(SOC18[t] <= SOC_max)
    
    # Final SOC constraints
    constraints.append(SOC10[T] >= SOC_min)
    constraints.append(SOC10[T] <= SOC_max)
    constraints.append(SOC18[T] >= SOC_min)
    constraints.append(SOC18[T] <= SOC_max)
    
    # End-of-day SOC must be >= initial SOC (battery cycling constraint)
    constraints.append(SOC10[T] >= SOC_0)
    constraints.append(SOC18[T] >= SOC_0)
    
    # SOLVE OPTIMIZATION
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    
    if problem.status == cp.OPTIMAL:
        print(f"Optimization successful! Total cost: ${problem.value:.2f}")
        
        # Return results
        return {
            'status': 'optimal',
            'total_cost': problem.value,
            'P_import': P_import.value,
            'P_export': P_export.value,
            'P_net_zone10': P_net_zone10.value,
            'P_net_zone18': P_net_zone18.value,
            'P_bat10_dis': P_bat10_dis.value,
            'P_bat10_chg': P_bat10_chg.value,
            'SOC10': SOC10.value,
            'P_bat18_dis': P_bat18_dis.value,
            'P_bat18_chg': P_bat18_chg.value,
            'SOC18': SOC18.value,
        }
    else:
        print(f"Optimization failed with status: {problem.status}")
        return {'status': 'failed'}


def export_results_to_excel(results, data_loader, scenario_name=None, output_dir='venv/data/optimization_results'):
    """
    Export optimization results to Excel file for power balance verification
    
    Args:
        results: Dictionary from optimize_two_battery_system()
        data_loader: DataLoader instance with input data
        output_dir: Directory to save results
    """
    from pathlib import Path
    import os
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create time index (0 to T-1 hours)
    hours = np.arange(data_loader.T)
    
    # Prepare data for Excel export
    # Main results DataFrame
    results_df = pd.DataFrame({
        'Hour': hours,
        
        # Load data per zone
        'Load_Zone10_kW': data_loader.p_load_zone10,
        'Load_Zone18_kW': data_loader.p_load_zone18,
        'Total_Load_kW': data_loader.p_load_zone10 + data_loader.p_load_zone18,
        
        # PV generation per zone  
        'PV_Zone10_kW': data_loader.pv_gen_zone10,
        'PV_Zone18_kW': data_loader.pv_gen_zone18,
        'Total_PV_kW': data_loader.total_pv_generation,
        
        # Battery schedules Zone 10
        'Bat10_Discharge_kW': results['P_bat10_dis'],
        'Bat10_Charge_kW': results['P_bat10_chg'],
                   
        'Bat10_Net_kW':  results['P_bat10_chg'] - results['P_bat10_dis'],
        'SOC10_percent': results['SOC10'][:-1],  # Exclude final SOC value
        
        # Battery schedules Zone 18
        'Bat18_Discharge_kW': results['P_bat18_dis'],
        'Bat18_Charge_kW': results['P_bat18_chg'],
                    
        'Bat18_Net_kW':  results['P_bat18_chg'] - results['P_bat18_dis'],
        'SOC18_percent': results['SOC18'][:-1],  # Exclude final SOC value
        
        # Grid power flows
        'Grid_Import_kW': results['P_import'],
        'Grid_Export_kW': results['P_export'],
        'Grid_Net_kW': results['P_import'] - results['P_export'],
        
        # Zone net power (what each zone needs from/gives to grid)
        'Zone10_Net_Power_kW': results['P_net_zone10'],
        'Zone18_Net_Power_kW': results['P_net_zone18'],
        'Total_Zone_Net_kW': results['P_net_zone10'] + results['P_net_zone18'],
    })
    
    # Power balance verification columns
    results_df['Zone10_Balance_Check'] = (
        results_df['Zone10_Net_Power_kW'] + 
        results_df['PV_Zone10_kW'] - 
        results_df['Bat10_Net_kW'] - 
        results_df['Load_Zone10_kW']
    )
    
    results_df['Zone18_Balance_Check'] = (
        results_df['Zone18_Net_Power_kW'] + 
        results_df['PV_Zone18_kW'] - 
        results_df['Bat18_Net_kW'] - 
        results_df['Load_Zone18_kW']
    )
    
    results_df['Grid_Balance_Check'] = (
        results_df['Grid_Net_kW'] - results_df['Total_Zone_Net_kW']
    )
    
    # Round all numeric columns to 3 decimal places
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(3)
    
    # Save to Excel - single sheet with detailed hourly results
    if scenario_name:
        filename = output_path / f'{scenario_name}_optimization_results.xlsx'
    else:
        filename = output_path / 'battery_optimization_results.xlsx'
    results_df.to_excel(filename, index=False)
    
    print(f"Results exported to: {filename}")
    print(f"Max power balance error: {np.max([np.max(np.abs(results_df['Zone10_Balance_Check'])), np.max(np.abs(results_df['Zone18_Balance_Check'])), np.max(np.abs(results_df['Grid_Balance_Check']))]):.6f} kW")
    
    return filename


# USAGE EXAMPLE
def generate_all_scenarios():
    """Generate optimization results for all standalone scenarios"""
    scenarios = [
        {'season': 'summer', 'marae': True, 'name': 'summer_marae'},
        {'season': 'summer', 'marae': False, 'name': 'summer_no_marae'},
        {'season': 'winter', 'marae': True, 'name': 'winter_marae'},
        {'season': 'winter', 'marae': False, 'name': 'winter_no_marae'}
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Generating optimization for: {scenario['name']}")
        print(f"{'='*50}")
        
        # Load data for this scenario
        loader = DataLoader()
        loader.load_profiles(season=scenario['season'], marae=scenario['marae'], C_pv10=18, C_pv18=18)
        
        # Optimize
        results = optimize_two_battery_system(loader)
        
        if results['status'] == 'optimal':
            print(f"Total daily cost: ${results['total_cost']:.2f}")
            print(f"Max SOC Zone10: {np.max(results['SOC10']):.1f}%")
            print(f"Max SOC Zone18: {np.max(results['SOC18']):.1f}%")
            
            # Export results to Excel with scenario name
            export_results_to_excel(results, loader, scenario['name'])
            print(f"✅ Completed: {scenario['name']}")
        else:
            print(f"❌ Optimization failed for: {scenario['name']}")


if __name__ == "__main__":
    # Uncomment the line below to generate all scenarios at once
    generate_all_scenarios()
    
    # Or run a single scenario:
    # Load data
    loader = DataLoader()
    loader.load_profiles(season='summer', marae=False, C_pv10=18, C_pv18=18)
    
    # Define scenario name based on parameters
    scenario_name = 'summer_no_marae'
    
    # Optimize
    results = optimize_two_battery_system(loader)
    
    if results['status'] == 'optimal':
        print(f"Total daily cost: ${results['total_cost']:.2f}")
        print(f"Max SOC Zone10: {np.max(results['SOC10']):.1f}%")
        print(f"Max SOC Zone18: {np.max(results['SOC18']):.1f}%")
        
        # Export results to Excel with scenario name
        export_results_to_excel(results, loader, scenario_name)
    else:
        print("Optimization failed - no results to export")