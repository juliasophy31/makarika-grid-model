# visualization/network_map.py
# Creates interactive geographic network visualization used for streamlit app

#requires: folium, pandas, pandapower

import folium
from folium import plugins
import pandas as pd

class NetworkVisualizer:
    """Creates interactive map visualizations of the network"""
    
    def __init__(self, network_manager):
        self.network_manager = network_manager
        self.net = network_manager.current_net
        self.buses_df = network_manager.buses_df
    
    def create_interactive_map(self, show_results=False, results_data=None):
        """
        Create an interactive Folium map of the network
        
        Args:
            show_results: If True, color components based on simulation results
            results_data: Dict with simulation results (power flows, voltages, etc.)
        
        Returns:
            folium.Map object
        """
        # Calculate center point
        center_lat = self.buses_df["y_coord"].mean()
        center_lon = self.buses_df["x_coord"].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="OpenStreetMap"
        )
        
        # Add layers
        self._add_lines(m, show_results, results_data)
        self._add_buses(m, show_results, results_data)
        self._add_loads(m)
        self._add_pv_systems(m, show_results, results_data)
        self._add_batteries(m, show_results, results_data)
        self._add_grid_connection(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add legend
        self._add_legend(m, show_results)
        
        return m
    
    def _add_lines(self, m, show_results=False, results_data=None):
        """Add transmission lines to map"""
        line_layer = folium.FeatureGroup(name="Lines", show=True)
        
        for idx, line in self.net.line.iterrows():
            # Get bus coordinates
            from_bus_name = self.net.bus.at[line['from_bus'], 'name']
            to_bus_name = self.net.bus.at[line['to_bus'], 'name']
            
            from_bus = self.buses_df[self.buses_df["bus_id"] == from_bus_name].iloc[0]
            to_bus = self.buses_df[self.buses_df["bus_id"] == to_bus_name].iloc[0]
            
            coords = [
                [from_bus["y_coord"], from_bus["x_coord"]],
                [to_bus["y_coord"], to_bus["x_coord"]]
            ]
            
            # Determine line status and color
            if not line['in_service']:
                color = 'red'
                weight = 3
                opacity = 0.5
                status = "❌ DISCONNECTED"
            else:
                # Color by voltage level
                if from_bus["vn_kv"] >= 10:
                    color = 'green'
                    weight = 5
                else:
                    color = 'blue'
                    weight = 3
                opacity = 0.8
                status = "✅ Connected"
            
            # Create popup
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <h4 style='color: #2C3E50; margin-bottom: 10px;'>{line['name']}</h4>
                <p><strong>From:</strong> {from_bus_name}</p>
                <p><strong>To:</strong> {to_bus_name}</p>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Length:</strong> {line['length_km']:.3f} km</p>
                <p><strong>Type:</strong> {line.get('std_type', 'Standard')}</p>
            """
            
            # Add loading information if results available
            if show_results and results_data and 'line_loading' in results_data:
                if idx in results_data['line_loading']:
                    loading = results_data['line_loading'][idx]
                    popup_html += f"<p><strong>Loading:</strong> {loading:.1f}%</p>"
            
            popup_html += "</div>"
            
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=opacity,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(line_layer)
        
        line_layer.add_to(m)
    
    def _add_buses(self, m, show_results=False, results_data=None):
        """Add bus markers to map"""
        bus_layer = folium.FeatureGroup(name="Buses", show=True)
        
        for idx, row in self.buses_df.iterrows():
            bus_idx = self.network_manager.bus_map[row["bus_id"]]
            
            # Determine color based on voltage level or results
            if show_results and results_data and 'bus_voltage' in results_data:
                if bus_idx in results_data['bus_voltage']:
                    voltage_pu = results_data['bus_voltage'][bus_idx]
                    if voltage_pu < 0.95:
                        color = 'red'
                        status = "⚠️ Low Voltage"
                    elif voltage_pu > 1.05:
                        color = 'orange'
                        status = "⚠️ High Voltage"
                    else:
                        color = 'green'
                        status = "✅ Normal"
                else:
                    color = 'gray'
                    status = "No data"
            else:
                # Color by voltage level
                if row["vn_kv"] >= 10:
                    color = 'green'
                else:
                    color = 'blue'
                status = "Operational"
            
            # Create popup
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <h4 style='color: #2E86C1;'>{row['bus_id']}</h4>
                <p><strong>Voltage Level:</strong> {row['vn_kv']} kV</p>
            """
            
            if show_results and results_data and 'bus_voltage' in results_data:
                if bus_idx in results_data['bus_voltage']:
                    voltage_pu = results_data['bus_voltage'][bus_idx]
                    voltage_kv = voltage_pu * row['vn_kv']
                    popup_html += f"""
                    <p><strong>Voltage:</strong> {voltage_pu:.3f} p.u. ({voltage_kv:.2f} kV)</p>
                    <p><strong>Status:</strong> {status}</p>
                    """
            
            popup_html += "</div>"
            
            folium.CircleMarker(
                location=[row["y_coord"], row["x_coord"]],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fill_opacity=0.7,
                weight=2
            ).add_to(bus_layer)
        
        bus_layer.add_to(m)
    
    def _add_loads(self, m):
        """Add load markers to map"""
        load_layer = folium.FeatureGroup(name="Loads", show=True)
        
        for idx, load in self.net.load.iterrows():
            if not load['in_service']:
                continue
            
            bus_name = self.net.bus.at[load['bus'], 'name']
            bus_data = self.buses_df[self.buses_df["bus_id"] == bus_name].iloc[0]
            
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <h4 style='color: #E67E22;'>{load['name']}</h4>
                <p><strong>Bus:</strong> {bus_name}</p>
                <p><strong>Power:</strong> {load['p_mw']*1000:.1f} kW</p>
            </div>
            """
            
            folium.Marker(
                location=[bus_data["y_coord"], bus_data["x_coord"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='orange', icon='home', prefix='fa')
            ).add_to(load_layer)
        
        load_layer.add_to(m)
    
    def _add_pv_systems(self, m, show_results=False, results_data=None):
        """Add PV system markers to map"""
        pv_layer = folium.FeatureGroup(name="PV Systems", show=True)
        
        for idx, pv in self.net.sgen.iterrows():
            bus_name = self.net.bus.at[pv['bus'], 'name']
            bus_data = self.buses_df[self.buses_df["bus_id"] == bus_name].iloc[0]
            
            # Determine icon color based on status
            if not pv['in_service']:
                icon_color = 'gray'
                status = "❌ Disabled"
            else:
                icon_color = 'orange'  # 'yellow' is not a valid color in Folium
                status = "✅ Active"
            
            popup_html = f"""
            <div style='width: 300px; font-family: Arial;'>
                <h4 style='color: #F39C12;'>{pv['name']}</h4>
                <p><strong>Bus:</strong> {bus_name}</p>
                <p><strong>Capacity:</strong> {pv['p_mw']*1000:.0f} kW</p>
                <p><strong>Status:</strong> {status}</p>
            """
            
            # Add generation data if available
            if show_results and results_data and 'pv_generation' in results_data:
                if pv['name'] in results_data['pv_generation']:
                    gen_data = results_data['pv_generation'][pv['name']]
                    popup_html += f"""
                    <p><strong>Current Gen:</strong> {gen_data['current']:.2f} kW</p>
                    <p><strong>Peak Gen:</strong> {gen_data['peak']:.2f} kW</p>
                    <p><strong>Total Energy:</strong> {gen_data['total']:.1f} kWh</p>
                    """
            
            popup_html += "</div>"
            
            folium.Marker(
                location=[bus_data["y_coord"], bus_data["x_coord"]],
                popup=folium.Popup(popup_html, max_width=350),
                icon=folium.Icon(color=icon_color, icon='sun', prefix='fa')
            ).add_to(pv_layer)
        
        pv_layer.add_to(m)
    
    def _add_batteries(self, m, show_results=False, results_data=None):
        """Add battery markers to map"""
        if not hasattr(self.net, 'storage') or len(self.net.storage) == 0:
            return
        
        battery_layer = folium.FeatureGroup(name="Batteries", show=True)
        
        for idx, battery in self.net.storage.iterrows():
            bus_name = self.net.bus.at[battery['bus'], 'name']
            bus_data = self.buses_df[self.buses_df["bus_id"] == bus_name].iloc[0]
            
            # Get initial SOC (default 50% if not available)
            initial_soc = 50.0
            if show_results and results_data and 'battery_status' in results_data:
                if battery['name'] in results_data['battery_status']:
                    initial_soc = results_data['battery_status'][battery['name']].get('initial_soc', 50.0)
            
            popup_html = f"""
            <div style='width: 350px; font-family: Arial;'>
                <h4 style='color: #8E44AD;'>🔋 {battery['name']}</h4>
                <p><strong>Bus:</strong> {bus_name}</p>
                <p><strong>Capacity:</strong> {battery['max_e_mwh']*1000:.0f} kWh</p>
                <p><strong>Power Rating:</strong> ±{battery['max_p_mw']*1000:.0f} kW</p>
                <p><strong>Initial SOC:</strong> {initial_soc:.1f}%</p>
            """
            
            # Add operational data if available
            if show_results and results_data and 'battery_status' in results_data:
                if battery['name'] in results_data['battery_status']:
                    batt_data = results_data['battery_status'][battery['name']]
                    
                    # Determine status color
                    if batt_data['power'] > 1:
                        status = "⚡ Discharging"
                        status_color = "#E74C3C"
                    elif batt_data['power'] < -1:
                        status = "🔌 Charging"
                        status_color = "#27AE60"
                    else:
                        status = "⏸️ Standby"
                        status_color = "#F39C12"
                    
                    popup_html += f"""
                    <div style='background-color: #F8F9FA; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <p style='color: {status_color}; font-weight: bold;'>{status}</p>
                        <p><strong>Power:</strong> {batt_data['power']:.2f} kW</p>
                        <p><strong>SOC:</strong> {batt_data['soc']:.1f}%</p>
                        <p><strong>Cycles:</strong> {batt_data['cycles']:.2f}</p>
                    </div>
                    """
            
            popup_html += "</div>"
            
            folium.Marker(
                location=[bus_data["y_coord"], bus_data["x_coord"]],
                popup=folium.Popup(popup_html, max_width=400),
                icon=folium.Icon(color='purple', icon='battery-half', prefix='fa')
            ).add_to(battery_layer)
        
        battery_layer.add_to(m)
    
    def _add_grid_connection(self, m):
        """Add external grid connection marker"""
        grid_layer = folium.FeatureGroup(name="Grid Connection", show=True)
        
        if len(self.net.ext_grid) > 0:
            grid = self.net.ext_grid.iloc[0]
            bus_name = self.net.bus.at[grid['bus'], 'name']
            bus_data = self.buses_df[self.buses_df["bus_id"] == bus_name].iloc[0]
            
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <h4 style='color: #2C3E50;'>🔌 Grid Connection</h4>
                <p><strong>Connection Point:</strong> {bus_name}</p>
                <p><strong>Voltage:</strong> {bus_data['vn_kv']} kV</p>
                <p><strong>Type:</strong> External Grid</p>
            </div>
            """
            
            folium.Marker(
                location=[bus_data["y_coord"], bus_data["x_coord"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='black', icon='plug', prefix='fa')
            ).add_to(grid_layer)
        
        grid_layer.add_to(m)
    
    def _add_legend(self, m, show_results=False):
        """Add legend to map"""
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 220px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px;
                    border-radius: 10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0; color: #2C3E50;">Network Legend</h4>
        <p><span style="color: green;">━━</span> MV Lines (11kV)</p>
        <p><span style="color: blue;">━━</span> LV Lines (0.4kV)</p>
        <p><span style="color: red;">━━</span> Disconnected Lines</p>
        <p><span style="color: darkred;">●</span> MV Buses</p>
        <p><span style="color: blue;">●</span> LV Buses</p>
        '''
        
        if show_results:
            legend_html += '''
            <hr style="margin: 10px 0;">
            <h5 style="margin: 5px 0; color: #2C3E50;">Status Colors:</h5>
            <p><span style="color: green;">●</span> Normal</p>
            <p><span style="color: orange;">●</span> Warning</p>
            <p><span style="color: red;">●</span> Critical</p>
            '''
        
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))