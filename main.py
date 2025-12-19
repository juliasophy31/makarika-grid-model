# main.py - Main entry point to run single microgrid simulations
#adjust scenarios in scenario_definitions file, choose scenario here

from pathlib import Path
from models.network_manager import NetworkManager
from scenarios.scenario_manager import ScenarioManager
from simulation.simulator import GridSimulator

def main():
    """
    Run microgrid simulation

    Available scenarios:
        Local Control:
        - 'local_winter_no_marae' - Local control, winter profiles, no marae event
        - 'local_winter_marae' - Local control, winter profiles, with marae event
        - 'local_summer_no_marae' - Local control, summer profiles, no marae event
        - 'local_summer_marae' - Local control, summer profiles, with marae event

        Time-of-Use Control:
        - 'tou_winter_no_marae' - ToU control, winter profiles, no marae event
        - 'tou_winter_marae' - ToU control, winter profiles, with marae event
        - 'tou_summer_no_marae' - ToU control, summer profiles, no marae event
        - 'tou_summer_marae' - ToU control, summer profiles, with marae event
        Optimized Control:
        - 'optimized_winter_no_marae' - Optimized control, winter profiles, no marae event
        - 'optimized_winter_marae' - Optimized control, winter profiles, with marae event
        - 'optimized_summer_no_marae' - Optimized control, summer profiles, no marae event
        - 'optimized_summer_marae' - Optimized control, summer profiles, with marae event
    """

    # ============================================
    # CHOOSE SCENARIO HERE
    # ============================================
    scenario_name = 'optimized_summer_no_marae'  # Choose from scenarios above
    # ============================================

    print("MICROGRID SIMULATOR")

    # Setup paths
    data_dir = Path("venv/data/")
    config_path = Path("config/scenario_definitions.yaml")

    # Initialize components
    print("\n🔧 Initializing simulation framework...\n")

    # 1. Network Manager - handles topology
    network_mgr = NetworkManager(data_dir)
    network_mgr.build_base_network()

    # 2. Scenario Manager - handles configurations
    scenario_mgr = ScenarioManager(config_path, network_mgr, data_dir)

    # 3. Simulator - runs everything
    simulator = GridSimulator(network_mgr, scenario_mgr, data_dir)

    # Run the specified scenario
    print("\n" + "="*60)
    print(f"RUNNING SCENARIO: {scenario_name}")
    print("="*60)

    results = simulator.run_scenario(scenario_name, n_steps=24)

    print("\n✅ Simulation complete!")

    return simulator, results

if __name__ == "__main__":
    simulator, results = main()