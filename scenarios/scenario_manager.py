# scenarios/scenario_manager.py
"""
ScenarioManager: Load Profile and Battery Control Configuration
================================================================

This module manages simulation scenarios by loading and applying different 
configurations for load profiles, PV generation, battery control strategies,
and market pricing.

What it does:
-------------
- Reads scenario definitions from scenario_definitions.yaml
- Loads load and PV generation time series data
- Applies battery control strategies (local, time-of-use, or optimized)
- Configures market pricing (peak/off-peak rates, export tariffs)
- Handles marae event schedules (increased loads during gatherings)

How to use:
-----------
1. Initialize with paths to config file, network manager, and data directory:

The manager automatically:
- Loads CSV profiles for loads and PV generation
- Creates and configures battery controllers
- Sets up time-series data sources for pandapower
- Applies event schedules if specified

Dependencies:
-------------
- scenario_definitions.yaml: Scenario configuration file
- CSV data files: Load profiles and PV capacity factors
- ControllerFactory: Creates battery controller instances
"""

import yaml
import pandas as pd
from pathlib import Path
from pandapower.control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from controllers.controller_factory import ControllerFactory

class ScenarioManager:
    """Orchestrates different control scenarios and configurations"""
    
    def __init__(self, config_path, network_manager, data_dir):
        self.config_path = Path(config_path)
        self.network_manager = network_manager
        self.data_dir = Path(data_dir)
        self.scenarios = {}
        self.active_scenario = None
        self.controllers = []
        self.market_config = {}  # Global market configuration

        self._load_scenarios()
    
    def _load_scenarios(self):
        """Load scenario definitions from YAML including defaults"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

            # Load defaults section
            self.defaults = config.get('defaults', {})
            self.market_config = self.defaults.get('market', {})  # Shortcut access

            # Load scenarios
            self.scenarios = config.get('scenarios', {})

        print(f"📋 Loaded {len(self.scenarios)} scenarios:")
        for name in self.scenarios.keys():
            print(f"  - {name}")
    
    def apply_scenario(self, scenario_name):
        """Apply a specific scenario configuration"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        net = self.network_manager.current_net
        
        print(f"\n🎬 Applying scenario: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Clear existing controllers
        self._clear_controllers(net)
        
        # Setup PV systems with time-series profiles
        self._setup_pv_systems(net, scenario.get('pv_systems', []))
        
        # Setup load profiles
        self._setup_loads(net, scenario.get('loads', {}))
        
        # Setup battery storage systems
        self._setup_batteries(net, scenario.get('batteries', []))
        
        # Setup controllers
        self._setup_controllers(net, scenario)
        
        self.active_scenario = scenario_name
        print(f"✅ Scenario '{scenario_name}' applied successfully\n")
        
        return net
    
    def _setup_pv_systems(self, net, pv_configs):
        """Setup PV systems with time-series profiles using standard profile"""
        # Handle both dict and list formats for backward compatibility
        if isinstance(pv_configs, dict):
            # New format: single dict with profile_file and profile_column
            profile_file = self.data_dir / pv_configs.get('profile_file', 'pv_standard_profile.csv')
            profile_column = pv_configs.get('profile_column', 'capacity_factor')
            delimiter = pv_configs.get('delimiter', ';')
        elif isinstance(pv_configs, list) and len(pv_configs) > 0:
            # Legacy format: list of PV configs
            profile_file = self.data_dir / pv_configs[0].get('profile_file', 'pv_standard_profile.csv')
            profile_column = pv_configs[0].get('profile_column', 'capacity_factor')
            delimiter = pv_configs[0].get('delimiter', ';')
        else:
            # Default fallback
            profile_file = self.data_dir / 'pv_standard_profile.csv'
            profile_column = 'capacity_factor'
            delimiter = ';'

        if not profile_file.exists():
            print(f"⚠️ Warning: PV profile not found at {profile_file}")
            return

        df_standard = pd.read_csv(profile_file, delimiter=delimiter, index_col=0)

        # Select the appropriate capacity factor column (e.g., capacity_factor_winter or capacity_factor_summer)
        if profile_column not in df_standard.columns:
            print(f"⚠️ Warning: Column '{profile_column}' not found in {profile_file}. Available columns: {list(df_standard.columns)}")
            return

        capacity_factors = df_standard[profile_column]

        # Apply profile to ALL PV systems in the network
        pv_profile_dict = {}
        pv_indices = []
        profile_names = []

        for idx, pv_row in net.sgen.iterrows():
            if not pv_row['in_service']:
                continue

            pv_name = pv_row['name']
            capacity_mw = pv_row['p_mw']

            # Scale standard profile by PV capacity (convert kW profile to MW for pandapower)
            # Note: capacity_factors are normalized 0-1, multiplied by capacity in MW
            pv_profile_dict[pv_name] = capacity_factors * capacity_mw
            pv_indices.append(idx)
            profile_names.append(pv_name)

        # Create time-series control for all PV systems
        if pv_profile_dict:
            df_pv = pd.DataFrame(pv_profile_dict)

            ConstControl(net, "sgen", "p_mw",
                        element_index=pv_indices,
                        profile_name=profile_names,
                        data_source=DFData(df_pv))

            print(f"☀️ Configured {len(pv_indices)} PV systems with profile column '{profile_column}'")
    
    def _setup_loads(self, net, load_config):
        """Setup load profiles from CSV or Excel files"""
        if not load_config:
            return

        profile_file = self.data_dir / 'loads_profile.xlsx'
        if not profile_file.exists():
            print(f"⚠️ Load profile {profile_file} not found")
            return

        # Handle Excel or CSV files
        file_ext = profile_file.suffix.lower()

        if file_ext in ['.xlsx', '.xls']:
            # Excel file - use sheet_name parameter
            sheet_name = load_config.get('sheet_name', 0)  # Default to first sheet if not specified
            df_loads = pd.read_excel(profile_file, sheet_name=sheet_name, index_col=0)
         
        else:
            # CSV file - use delimiter parameter
            delimiter = load_config.get('delimiter', ',')
            df_loads = pd.read_csv(profile_file, index_col=0, delimiter=delimiter)
    

        # Convert load profiles from kW to MW for pandapower
        df_loads = df_loads / 1000.0

        # Filter enabled loads
        enabled_loads = load_config.get('enabled_loads')
        if enabled_loads:
            available_cols = [col for col in enabled_loads if col in df_loads.columns]
            df_loads = df_loads[available_cols]

        # Map load names to indices
        load_indices = []
        profile_names = []

        for load_name in df_loads.columns:
            load_match = net.load[net.load.name == load_name]
            if not load_match.empty:
                load_indices.append(load_match.index[0])
                profile_names.append(load_name)

        if load_indices:
            ConstControl(net, "load", "p_mw",
                        element_index=load_indices,
                        profile_name=profile_names,
                        data_source=DFData(df_loads))


    
    def _setup_batteries(self, net, battery_configs):
        """Create battery storage systems"""
        if not battery_configs:
            return
        
        for batt_config in battery_configs:
            bus_name = batt_config['bus']
            battery_name = batt_config['name']
            
            # Check if battery already exists
            existing = net.storage[net.storage.name == battery_name]
            
            if existing.empty:
                # Get bus index
                bus_match = net.bus[net.bus.name == bus_name]
                if bus_match.empty:
                    print(f"⚠️ Warning: Bus {bus_name} not found for battery {battery_name}")
                    continue
                
                bus_idx = bus_match.index[0]
                
                # Create battery
                import pandapower as pp
                pp.create_storage(
                    net,
                    bus=bus_idx,
                    p_mw=0.0,
                    max_e_mwh=batt_config['max_e_mwh'],
                    soc_percent=batt_config.get('soc_initial', 50),
                    max_p_mw=batt_config['max_p_mw'],
                    min_p_mw=-batt_config['max_p_mw'],
                    efficiency_percent=batt_config.get('efficiency', 0.9) * 100,
                    name=battery_name
                )
                
                print(f"🔋 Created battery: {battery_name} at {bus_name} "
                      f"({batt_config['max_e_mwh']*1000:.0f}kWh, ±{batt_config['max_p_mw']*1000:.0f}kW)")
   
    def _prepare_scenario_data(self, scenario):
        """
        Prepare load profiles, PV profiles, and pricing data for controllers
        
        CRITICAL FIX: Converts PV capacity factors to actual kW values
        with proper column names matching network PV system names
        
        Returns:
            dict with 'load_profile', 'pv_profile', 'prices'
        """
        # Load the load profile
        load_config = scenario.get('loads', {})
        profile_file = self.data_dir / 'loads_profile.xlsx'
        
        if profile_file.exists():
            sheet_name = load_config.get('sheet_name', 'winter_no_marae')
            load_profile = pd.read_excel(profile_file, sheet_name=sheet_name, index_col=0)
            print(f"   Loaded load profile from sheet: {sheet_name}")
        else:
            print(f"   Warning: Load profile not found at {profile_file}")
            load_profile = pd.DataFrame()
        
        # Load the PV profile 
        pv_config = scenario.get('pv_systems', {})
        pv_file = self.data_dir / 'pv_standard_profile.csv'
        delimiter = pv_config.get('delimiter', ';')
        profile_column = pv_config.get('profile_column', 'capacity_factor')
        
        if pv_file.exists():
            # Load capacity factors (0-1 range)
            pv_profile_raw = pd.read_csv(pv_file, delimiter=delimiter, index_col=0)
            
            # Extract the specific column for this scenario
            if profile_column in pv_profile_raw.columns:
                capacity_factors = pv_profile_raw[profile_column].values
            else:
                print(f"   Warning: Column '{profile_column}' not found, using first column")
                capacity_factors = pv_profile_raw.iloc[:, 0].values
            
            print(f"   Loaded PV capacity factors from column: {profile_column}")
            
            # Convert capacity factors to actual kW generation
            # Get PV system capacities from network
            net = self.network_manager.current_net
            
            pv_profile_kw = pd.DataFrame(index=pv_profile_raw.index)
            
            for idx, pv_row in net.sgen.iterrows():
                if not pv_row['in_service']:
                    continue
                    
                pv_name = pv_row['name']
                pv_capacity_kw = pv_row['p_mw'] * 1000  # Convert MW to kW
                
                # Multiply capacity factor by PV capacity to get actual generation
                pv_profile_kw[pv_name] = capacity_factors * pv_capacity_kw
                
                print(f"   Created PV profile for {pv_name}: {pv_capacity_kw:.1f} kW capacity")
            
            pv_profile = pv_profile_kw
            
            # Verify we have the expected columns
            expected_pv_names = ['PV_bus_10', 'PV_bus_18']
            missing = [name for name in expected_pv_names if name not in pv_profile.columns]
            if missing:
                print(f"   ⚠️  Warning: Expected PV systems not found: {missing}")
        else:
            print(f"   Warning: PV profile not found at {pv_file}")
            pv_profile = pd.DataFrame()
        
        # Prepare pricing data
        market_config = self.defaults.get('market', {})
        
        prices = {
            'peak_hours': market_config.get('peak_hours', [7, 8, 9, 10, 17, 18, 19, 20]),
            'import_peak': market_config.get('peak_import_price', 0.33),
            'import_offpeak': market_config.get('offpeak_import_price', 0.23),
            'export': market_config.get('export_price', 0.13)
        }
        
        print(f"   Pricing: Peak ${prices['import_peak']:.2f}/kWh, Off-peak ${prices['import_offpeak']:.2f}/kWh")
        
        # Verify data integrity before returning
        if not pv_profile.empty:
            print(f"   PV profile shape: {pv_profile.shape}")
            print(f"   PV profile columns: {list(pv_profile.columns)}")
            print(f"   Sample peak generation (hour 13): {pv_profile.iloc[13].sum():.2f} kW total")
        
        return {
            'load_profile': load_profile,
            'pv_profile': pv_profile,
            'prices': prices
     }
    
    def _setup_controllers(self, net, scenario):
        """Setup battery/DR controllers based on scenario"""
        controller_configs = scenario.get('controllers', [])
        battery_configs = scenario.get('batteries', [])
        
        # Create a battery config lookup
        battery_lookup = {b['name']: b for b in battery_configs}

        # get market config from defaults and merge with scenario-specific config
        market_config = self.defaults.get('market', {}).copy()
        if 'market' in scenario:
            market_config.update(scenario['market'])

        # Prepare scenario data for controllers that need it
        scenario_data = None
        if any(ctrl.get('type') in ['OptimizedBatteryController'] for ctrl in controller_configs):
            scenario_data = self._prepare_scenario_data(scenario)
            
        self.controllers = []
        
        for ctrl_config in controller_configs:
            battery_name = ctrl_config.get('battery_name')
            batt_config = battery_lookup.get(battery_name)

            # add market config to controller config for tou battery
            ctrl_config_with_market = ctrl_config.copy()
            ctrl_config_with_market['market_config'] = market_config

            controller = ControllerFactory.create(net, ctrl_config_with_market, batt_config, scenario_data=scenario_data)
            self.controllers.append(controller)
        
        print(f"🎮 Created {len(self.controllers)} controllers")
    
    def _clear_controllers(self, net):
        """Remove all existing controllers"""
        if hasattr(net, 'controller') and len(net.controller) > 0:
            net.controller.drop(net.controller.index, inplace=True)
        self.controllers = []
    
    def get_active_scenario_info(self):
        """Get information about currently active scenario"""
        if not self.active_scenario:
            return None
        
        scenario = self.scenarios[self.active_scenario]
        
        return {
            'name': scenario['name'],
            'description': scenario['description'],
            'controllers': len(scenario.get('controllers', [])),
            'pv_systems': len([pv for pv in scenario.get('pv_systems', []) if pv.get('enabled', True)]),
            'market_config': scenario.get('market', {})
        }
    def get_market_config(self):
        """Get global market configuration"""
        return self.defaults.get('market', {})
    
    def list_scenarios(self):
        """List all available scenarios"""
        return [
            {
                'id': name,
                'name': config['name'],
                'description': config['description']
            }
            for name, config in self.scenarios.items()
        ]
    
    def apply_scenario_with_custom_profiles(self, scenario_name, load_profiles=None, pv_profiles=None):
        """
        Apply scenario with custom load and PV profiles.
        This is used for stochastic analysis.
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        net = self.network_manager.current_net
        
        print(f"\n🎬 Applying scenario with custom profiles: {scenario['name']}")
        
        # Clear existing controllers
        self._clear_controllers(net)
        
        # Setup custom PV systems if profiles provided
        if pv_profiles is not None:
            self._setup_pv_systems_from_dataframe(net, scenario, pv_profiles)
        else:
            self._setup_pv_systems(net, scenario.get('pv_systems', []))
        
        # Setup custom loads if profiles provided  
        if load_profiles is not None:
            self._setup_loads_from_dataframe(net, scenario, load_profiles)
        else:
            self._setup_loads(net, scenario.get('loads', {}))
        
        # Setup battery storage systems and controllers
        self._setup_batteries(net, scenario.get('batteries', []))
        self._setup_controllers(net, scenario)
        
        self.active_scenario = scenario_name
        print(f"✅ Scenario '{scenario_name}' applied with custom profiles\n")
        
        return net
    
    def _setup_pv_systems_from_dataframe(self, net, scenario, pv_profiles):
        """Setup PV systems using custom profiles from dataframe"""
        pv_configs = scenario.get('pv_systems', [])
        
        # Determine the profile column to use
        if isinstance(pv_configs, dict):
            profile_column = pv_configs.get('profile_column', 'capacity_factor')
        elif isinstance(pv_configs, list) and len(pv_configs) > 0:
            profile_column = pv_configs[0].get('profile_column', 'capacity_factor')
        else:
            profile_column = 'capacity_factor'
        
        if profile_column not in pv_profiles.columns:
            print(f"⚠️ Warning: Column '{profile_column}' not found in custom PV profiles")
            return
        
        # Get capacity factors from custom profiles
        capacity_factors = pv_profiles[profile_column]
        
        # Apply profile to all PV systems in the network
        pv_profile_dict = {}
        pv_indices = []
        profile_names = []
        
        for idx, pv_row in net.sgen.iterrows():
            if not pv_row['in_service']:
                continue
            
            pv_name = pv_row['name']
            capacity_mw = pv_row['p_mw']
            
            # Apply custom capacity factors to this PV system
            pv_profile_dict[pv_name] = capacity_factors * capacity_mw
            pv_indices.append(idx)
            profile_names.append(pv_name)
        
        # Create time-series control for all PV systems
        if pv_profile_dict:
            df_pv = pd.DataFrame(pv_profile_dict)
            
            ConstControl(net, "sgen", "p_mw",
                        element_index=pv_indices,
                        profile_name=profile_names,
                        data_source=DFData(df_pv))
            
            print(f"🌞 Applied custom PV profiles to {len(pv_indices)} PV systems")
    
    def _setup_loads_from_dataframe(self, net, scenario, load_profiles):
        """Setup loads using custom profiles from dataframe"""
        load_config = scenario.get('loads', {})
        
        if load_profiles.empty:
            print("⚠️ No custom load profiles provided")
            return
        
        # Convert load profiles from kW to MW for pandapower
        df_loads = load_profiles / 1000.0
        
        # Filter enabled loads if specified
        enabled_loads = load_config.get('enabled_loads')
        if enabled_loads:
            available_cols = [col for col in enabled_loads if col in df_loads.columns]
            df_loads = df_loads[available_cols]
        
        # Map load names to pandapower indices
        load_indices = []
        profile_names = []
        
        for load_name in df_loads.columns:
            load_match = net.load[net.load.name == load_name]
            if not load_match.empty:
                load_indices.append(load_match.index[0])
                profile_names.append(load_name)
            else:
                print(f"⚠️ Load '{load_name}' not found in network")
        
        if load_indices:
            ConstControl(net, "load", "p_mw",
                        element_index=load_indices,
                        profile_name=profile_names,
                        data_source=DFData(df_loads))
            
            print(f"📊 Applied custom load profiles to {len(load_indices)} loads")