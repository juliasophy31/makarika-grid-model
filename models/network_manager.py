# models/network_manager.py
"""
Network Manager - Microgrid Topology Builder
=============================================

Constructs and manages the pandapower network model representing the 
Tairāwhiti microgrid topology with buses, lines, transformers, and loads.

What it does:
-------------
- Builds pandapower network from CSV data files
- Creates buses at 11 kV (MV) and 415 V (LV) levels
- Establishes lines with proper electrical parameters
- Places transformers (30 kVA and 50 kVA)
- Configures loads at residential connection points
- Manages component registry for dynamic modifications
- Provides network state management (base vs modified)

How to use:
-----------

    from models.network_manager import NetworkManager
    from pathlib import Path
    
    # Initialize with data directory
    network_mgr = NetworkManager(data_dir=Path("venv/data"))
    
    # Build base network
    network_mgr.build_base_network()
    
    # Access network
    net = network_mgr.current_net
    
    # Run power flow
    import pandapower as pp
    pp.runpp(net)

Input Data Files:
-----------------
Located in venv/data/:
- network_waipiro_buses.csv: Bus definitions with coordinates
- network_waipiro_lines.csv: Line connections
- network_waipiro_loads.csv: Load connection points

Integration:
------------
Used by:
- ScenarioManager: Applies scenario configurations
- GridSimulator: Runs time-series simulations
- NetworkVisualizer: Creates interactive maps
- main.py: Entry point for simulations

Note: some functions are redundant, kept for future extensions. For example, add more PV systems dynamically, or modify loads during runtime.
"""

import pandas as pd
import pandapower as pp
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import copy

class NetworkManager:
    """Manages network topology and component modifications"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.base_net = None
        self.current_net = None
        self.bus_map = {}
        self.component_registry = {}
        
        # Load network data files
        self.buses_df = None
        self.lines_df = None
        self.loads_df = None
    
    def build_base_network(self):
        """Build the baseline network from CSV files"""
        print("📡 Building base network...")
        
        # Load CSV files
        self.buses_df = pd.read_csv(self.data_dir / "network_waipiro_buses.csv", delimiter=';')
        self.lines_df = pd.read_csv(self.data_dir / "network_waipiro_lines.csv", delimiter=';')
        self.loads_df = pd.read_csv(self.data_dir / "network_waipiro_loads.csv", delimiter=';')
        
        net = pp.create_empty_network()
        
        # Create buses
        for _, row in self.buses_df.iterrows():
            bus_idx = pp.create_bus(
                net, 
                name=row["bus_id"],
                vn_kv=row["vn_kv"], 
                geodata=(row["x_coord"], row["y_coord"])
            )
            self.bus_map[row["bus_id"]] = bus_idx
        
        # Create transformers
        self._create_transformers(net)
        
        # Create external grid
        pp.create_ext_grid(net, bus=self.bus_map["bus_22mv"], name="Grid Connection")
        
        # Create lines
        for idx, row in self.lines_df.iterrows():
            if row["from_bus"] in self.bus_map and row["to_bus"] in self.bus_map:
                length_km = self._calculate_line_length(row["from_bus"], row["to_bus"])

                # All lines use NAYY 4x50 SE type
                line_type = "NAYY 4x50 SE"

                line_idx = pp.create_line(
                    net,
                    from_bus=self.bus_map[row["from_bus"]],
                    to_bus=self.bus_map[row["to_bus"]],
                    length_km=length_km,
                    std_type=line_type,
                    name=row.get("line_id", f"line_{idx}")
                )

                # Register line
                line_name = row.get("line_id", f"line_{idx}")
                self.component_registry[line_name] = {
                    'type': 'line',
                    'index': line_idx,
                    'enabled': True
                }
        
        # Create loads
        for _, row in self.loads_df.iterrows():
            if row["bus_id"] in self.bus_map:
                p_mw = float(row["power_kw"]) / 1000.0
                q_mvar = p_mw * 0.2
                
                load_idx = pp.create_load(
                    net, 
                    bus=self.bus_map[row["bus_id"]],
                    p_mw=p_mw, 
                    q_mvar=q_mvar, 
                    name=row["load_id"]
                )
                
                # Register load
                self.component_registry[row["load_id"]] = {
                    'type': 'load', 
                    'index': load_idx, 
                    'enabled': True,
                    'bus': row["bus_id"], 
                    'rated_power': float(row["power_kw"]),
                    'original_p_mw': p_mw
                }
        
        # Create PV generators (from your original code)
        pv_configs = [
            ("bus_10lv", 0.018, "PV_bus_10"),  # 18 kW
            ("bus_18lv", 0.018, "PV_bus_18")   # 18 kW
        ]
        
        for bus_name, p_mw, pv_name in pv_configs:
            if bus_name in self.bus_map:
                pv_idx = pp.create_sgen(
                    net, 
                    bus=self.bus_map[bus_name], 
                    p_mw=p_mw, 
                    name=pv_name
                )
                
                # Register PV
                self.component_registry[pv_name] = {
                    'type': 'pv', 
                    'index': pv_idx, 
                    'enabled': True,
                    'bus': bus_name, 
                    'capacity_kw': p_mw * 1000
                }
        
        self.base_net = net
        self.current_net = copy.deepcopy(net)
        
        summary = self.get_network_summary()
        print(f"✅ Network built: {summary['buses']} buses, {summary['lines']} lines, "
              f"{summary['loads']} loads, {summary['pv_systems']} PV systems")
        
        return self.current_net
    
    def add_pv_system(self, bus_name, capacity_kw, name=None):
        """Dynamically add a PV system"""
        if bus_name not in self.bus_map:
            raise ValueError(f"Bus {bus_name} not found")

        # Check if this bus already has a PV system
        bus_idx = self.bus_map[bus_name]
        existing_pv = self.current_net.sgen[self.current_net.sgen['bus'] == bus_idx]
        if not existing_pv.empty:
            existing_pv_names = existing_pv['name'].tolist()
            raise ValueError(f"Bus {bus_name} already has PV system(s): {', '.join(existing_pv_names)}. "
                           f"Use PV scaling to increase capacity instead of adding a new system.")

        pv_name = name or f"PV_{bus_name}"
        pv_idx = pp.create_sgen(
            self.current_net,
            bus=self.bus_map[bus_name],
            p_mw=capacity_kw / 1000.0,
            name=pv_name
        )

        self.component_registry[pv_name] = {
            'type': 'pv',
            'index': pv_idx,
            'enabled': True,
            'bus': bus_name,
            'capacity_kw': capacity_kw
        }

        # Ensure bus columns remain integer type
        self._fix_bus_dtypes()

        print(f"✅ Added {capacity_kw}kW PV at {bus_name}")
        return pv_idx
    
    def remove_pv_system(self, pv_name):
        """Remove a PV system"""
        if pv_name not in self.component_registry:
            raise ValueError(f"PV system {pv_name} not found")

        pv_info = self.component_registry[pv_name]
        self.current_net.sgen.drop(pv_info['index'], inplace=True)
        pv_info['enabled'] = False

        # Ensure bus columns remain integer type after removal
        self._fix_bus_dtypes()

        print(f"❌ Removed PV {pv_name}")
    
    def toggle_pv_system(self, pv_name, enabled):
        """Enable/disable PV without removing it"""
        if pv_name not in self.component_registry:
            raise ValueError(f"PV system {pv_name} not found")

        pv_info = self.component_registry[pv_name]
        self.current_net.sgen.at[pv_info['index'], 'in_service'] = enabled
        pv_info['enabled'] = enabled

        status = "enabled" if enabled else "disabled"
        print(f"🔄 PV {pv_name} {status}")

    def scale_pv_system(self, pv_name, scale_factor):
        """Scale PV capacity by a factor (e.g., 1.5 = 150% of original capacity)"""
        if pv_name not in self.component_registry:
            raise ValueError(f"PV system {pv_name} not found")

        pv_info = self.component_registry[pv_name]
        original_capacity = pv_info['capacity_kw']
        new_capacity = original_capacity * scale_factor
        new_p_mw = new_capacity / 1000.0

        self.current_net.sgen.at[pv_info['index'], 'p_mw'] = new_p_mw

        print(f"☀️ PV {pv_name} scaled by {scale_factor}x: {original_capacity:.0f}kW → {new_capacity:.0f}kW")
    
    def disconnect_line(self, line_name):
        """Disconnect a line to simulate faults or islanding"""
        if line_name not in self.component_registry:
            raise ValueError(f"Line {line_name} not found")

        line_info = self.component_registry[line_name]
        self.current_net.line.at[line_info['index'], 'in_service'] = False
        line_info['enabled'] = False

        # Ensure bus columns remain integer type for numba compatibility
        self._fix_bus_dtypes()

        print(f"⚠️ Line {line_name} disconnected")

    def reconnect_line(self, line_name):
        """Reconnect a previously disconnected line"""
        if line_name not in self.component_registry:
            raise ValueError(f"Line {line_name} not found")

        line_info = self.component_registry[line_name]
        self.current_net.line.at[line_info['index'], 'in_service'] = True
        line_info['enabled'] = True

        # Ensure bus columns remain integer type for numba compatibility
        self._fix_bus_dtypes()

        print(f"✅ Line {line_name} reconnected")

    def _fix_bus_dtypes(self):
        """Fix bus column dtypes to prevent numba errors"""
        net = self.current_net

        # Ensure all bus reference columns are integers
        # Use fillna or drop individual rows to avoid reassigning the entire DataFrame
        if 'bus' in net.load.columns and len(net.load) > 0:
            # Drop rows with NaN bus values
            invalid_rows = net.load[net.load['bus'].isna()].index
            if len(invalid_rows) > 0:
                net.load.drop(invalid_rows, inplace=True)
            if len(net.load) > 0:
                net.load['bus'] = net.load['bus'].astype('int64')

        if 'bus' in net.sgen.columns and len(net.sgen) > 0:
            invalid_rows = net.sgen[net.sgen['bus'].isna()].index
            if len(invalid_rows) > 0:
                net.sgen.drop(invalid_rows, inplace=True)
            if len(net.sgen) > 0:
                net.sgen['bus'] = net.sgen['bus'].astype('int64')

        if hasattr(net, 'storage') and 'bus' in net.storage.columns and len(net.storage) > 0:
            invalid_rows = net.storage[net.storage['bus'].isna()].index
            if len(invalid_rows) > 0:
                net.storage.drop(invalid_rows, inplace=True)
            if len(net.storage) > 0:
                net.storage['bus'] = net.storage['bus'].astype('int64')

        if 'from_bus' in net.line.columns and len(net.line) > 0:
            invalid_rows = net.line[net.line['from_bus'].isna()].index
            if len(invalid_rows) > 0:
                net.line.drop(invalid_rows, inplace=True)
            if len(net.line) > 0:
                net.line['from_bus'] = net.line['from_bus'].astype('int64')

        if 'to_bus' in net.line.columns and len(net.line) > 0:
            invalid_rows = net.line[net.line['to_bus'].isna()].index
            if len(invalid_rows) > 0:
                net.line.drop(invalid_rows, inplace=True)
            if len(net.line) > 0:
                net.line['to_bus'] = net.line['to_bus'].astype('int64')
    
    def scale_load(self, load_name, scale_factor):
        """Scale a load by a factor"""
        if load_name not in self.component_registry:
            raise ValueError(f"Load {load_name} not found")
        
        load_info = self.component_registry[load_name]
        original_p = load_info['original_p_mw']
        
        new_p = original_p * scale_factor
        self.current_net.load.at[load_info['index'], 'p_mw'] = new_p
        self.current_net.load.at[load_info['index'], 'q_mvar'] = new_p * 0.2
        
        print(f"📊 Load {load_name} scaled by {scale_factor}x")
    
    def reset_network(self):
        """Reset to original base network"""
        self.current_net = copy.deepcopy(self.base_net)

        # Reset component registry states
        for comp_name, comp_info in self.component_registry.items():
            if comp_info['type'] != 'pv' or comp_name in ['PV_bus_10', 'PV_bus_18']:
                comp_info['enabled'] = True

        # Ensure proper dtypes after reset
        self._fix_bus_dtypes()

        print("🔄 Network reset to baseline")
    
    def get_network_summary(self):
        """Return summary of current network state"""
        return {
            'buses': len(self.current_net.bus),
            'lines': len(self.current_net.line[self.current_net.line.in_service]),
            'loads': len(self.current_net.load[self.current_net.load.in_service]),
            'pv_systems': len(self.current_net.sgen[self.current_net.sgen.in_service]),
            'batteries': len(self.current_net.storage) if hasattr(self.current_net, 'storage') else 0
        }
    
    def _calculate_line_length(self, from_bus_name, to_bus_name):
        """Calculate line length from coordinates"""
        fb = self.buses_df[self.buses_df["bus_id"] == from_bus_name].iloc[0]
        tb = self.buses_df[self.buses_df["bus_id"] == to_bus_name].iloc[0]
        
        R = 6371.0  # Earth radius in km
        dlat = radians(tb["y_coord"] - fb["y_coord"])
        dlon = radians(tb["x_coord"] - fb["x_coord"])
        
        a = (sin(dlat/2)**2 + 
             cos(radians(fb["y_coord"])) * cos(radians(tb["y_coord"])) * sin(dlon/2)**2)
        distance = 2 * R * atan2(sqrt(a), sqrt(1 - a))
        
        return max(distance, 0.001)
    
    def _create_transformers(self, net):
        """Create transformers"""
        # trafo1 connects to bus_10lv (50 kVA), trafo2 connects to bus_18lv (30 kVA)
        trafo_configs = [
            ("bus_trafo1_mv", "bus_trafo1_lv", "trafo1", 0.05),  # 50 kVA for bus 10
            ("bus_trafo2_mv", "bus_trafo2_lv", "trafo2", 0.03)   # 30 kVA for bus 18
        ]
        
        for mv_bus, lv_bus, name, sn_mva in trafo_configs:
            if mv_bus in self.bus_map and lv_bus in self.bus_map:
                pp.create_transformer_from_parameters(
                    net, 
                    hv_bus=self.bus_map[mv_bus], 
                    lv_bus=self.bus_map[lv_bus],
                    sn_mva=sn_mva, vn_hv_kv=11.0, vn_lv_kv=0.4,
                    vk_percent=4.0, vkr_percent=1.7,
                    pfe_kw=0.13, i0_percent=0.24, 
                    name=name
                )