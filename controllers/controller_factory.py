# controllers/controller_factory.py
"""
Controller Factory - Centralized Battery Controller Creation
=============================================================

This factory class provides a standardized way to create and configure different
battery controller types without having to deal with individual controller 
initialization details.

What it does:
-------------
- Creates battery controller instances (local, TOU, optimized)
- Handles controller-specific configuration requirements
- Creates battery storage elements if they don't exist
- Ensures consistent controller setup across the simulation framework
- Maps controller types to their implementations


Supported Controller Types:
---------------------------
- 'SimpleBattery': Local voltage-based control (local_battery.py)
- 'TimeOfUseBattery': Price-responsive TOU control (time_of_use_battery.py)
- 'OptimizedBatteryController': Perfect foresight optimization (optimized_battery_controller.py)

Integration:
------------
This factory is used by:
- ScenarioManager: Creates controllers when applying scenarios
- main.py: Sets up controllers for single simulation runs
- Stochastic analysis scripts: Batch controller creation

Configuration Sources:
----------------------
Controller configs typically come from:
- scenario_definitions.yaml: Controller specifications per scenario
- ScenarioManager: Passes configs with market pricing
- Manual dictionaries: For custom setups

Dependencies:
-------------
- local_battery.SimpleBattery
- time_of_use_battery.TOUBattery
- optimized_battery_controller.OptimizedBatteryController
- pandapower (for storage creation)

"""

import pandas as pd
import pandapower as pp
from controllers.local_battery import SimpleBattery
from controllers.time_of_use_battery import TOUBattery
from controllers.optimized_battery_controller import OptimizedBatteryController


class ControllerFactory:
    """Factory for creating different controller types"""

    _controller_registry = {
        'SimpleBattery': SimpleBattery,
        'TimeOfUseBattery': TOUBattery,
        'OptimizedBatteryController': OptimizedBatteryController,
    }
    
    @classmethod
    def create(cls, net, controller_config, battery_config=None, scenario_data=None):
        """
        Create a controller from configuration
        
        Args:
            net: pandapower network
            controller_config: dict with controller configuration
            battery_config: dict with battery storage configuration (optional)
            scenario_data: dict with load/PV profiles and prices (optional)
        
        Returns:
            Controller instance
        """
        ctrl_type = controller_config.get('type')
        
        if ctrl_type not in cls._controller_registry:
            raise ValueError(f"Unknown controller type: {ctrl_type}")

        if ctrl_type == 'SimpleBattery':
            return cls._create_simple_battery(net, controller_config, battery_config)
        elif ctrl_type == 'TimeOfUseBattery':
            return cls._create_tou_battery(net, controller_config, battery_config)
        elif ctrl_type == 'OptimizedBatteryController':
            return cls._create_optimized_battery(net, controller_config, scenario_data)

        raise NotImplementedError(f"Creation logic for {ctrl_type} not implemented")
    
    @classmethod
    def _create_simple_battery(cls, net, ctrl_config, batt_config):
        """Create SimpleBattery controller"""
        
        bus_name = ctrl_config['bus']
        battery_name = ctrl_config['battery_name']
        pv_name = ctrl_config['pv_name']
        local_loads = ctrl_config['local_loads']
        bus_label = ctrl_config.get('bus_label', bus_name)
        
        # Check if battery already exists
        existing_battery = net.storage[net.storage.name == battery_name]
        
        if existing_battery.empty and batt_config:
            # Create battery storage from config
            bus_idx = net.bus[net.bus.name == bus_name].index[0]
            
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
            
            print(f"🔋 Created battery storage: {battery_name} at {bus_name}")
        
        # Create controller
        controller = SimpleBattery(
            net,
            pv_name=pv_name,
            battery_name=battery_name,
            load_names=local_loads,
            bus_name=bus_label
        )
        
        return controller

    @classmethod
    def _create_tou_battery(cls, net, ctrl_config, batt_config):
        """Create TimeOfUseBattery controller"""

        bus_name = ctrl_config['bus']
        battery_name = ctrl_config['battery_name']
        pv_name = ctrl_config['pv_name']
        local_loads = ctrl_config.get('local_loads', None)  # NEW: get local loads
        bus_label = ctrl_config.get('bus_label', bus_name)

        # Get market configuration from scenario manager
        market_config = ctrl_config.get('market_config', {})

        # Check if battery already exists
        existing_battery = net.storage[net.storage.name == battery_name]

        if existing_battery.empty and batt_config:
            # Create battery storage from config
            bus_idx = net.bus[net.bus.name == bus_name].index[0]

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

            print(f"🔋 Created battery storage: {battery_name} at {bus_name}")

        # Create controller
        controller = TOUBattery(
            net,
            pv_name=pv_name,
            battery_name=battery_name,
            load_names=local_loads,  # NEW: pass local loads
            bus_name=bus_label,
            market_config=market_config
        )

        return controller
    
    @classmethod
    def _create_optimized_battery(cls, net, ctrl_config, scenario_data):
        """Create OptimizedBatteryController that applies pre-computed schedules"""
        
        # Get battery names
        battery1_name = ctrl_config.get('battery1_name', 'Battery_bus10')
        battery2_name = ctrl_config.get('battery2_name', 'Battery_bus18')
        
        # Get schedule file path
        schedule_file = ctrl_config.get('schedule_file', 'standalone_battery_schedules.csv')
        
        # Ensure both batteries exist in the network
        for battery_name in [battery1_name, battery2_name]:
            existing_battery = net.storage[net.storage.name == battery_name]
            if existing_battery.empty:
                raise ValueError(f"Battery {battery_name} not found in network. Create batteries first.")
        
        # Create simple schedule-based controller
        controller = OptimizedBatteryController(
            net=net,
            battery1_name=battery1_name,
            battery2_name=battery2_name,
            schedule_file=schedule_file
        )

        return controller


@classmethod
def register_controller(cls, name, controller_class):
    """Register a new controller type"""
    cls._controller_registry[name] = controller_class
    print(f" Registered controller type: {name}")