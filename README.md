# Makarika Microgrid Simulation

A comprehensive simulation framework for analyzing battery energy storage systems in the Tairāwhiti microgrid network, New Zealand.

## 🎯 Project Overview

This project compares three battery control strategies across different seasonal and operational conditions:

- **Local Control**: Autonomous voltage-based battery operation
- **Time-of-Use (TOU)**: Price-responsive charging/discharging
- **Optimized Control**: Perfect foresight mathematical optimization

## 🌟 Key Features

- ⚡ **Real-time Power Flow Analysis** using pandapower
- 📊 **Interactive Dashboard** with Streamlit
- 🎲 **Monte Carlo Stochastic Analysis** (±10% load, ±20% PV variation)
- 🗺️ **Network Visualization** with interactive Folium maps
- 💰 **Economic Analysis** with Time-of-Use tariffs
- 🔋 **Battery Control Algorithms** (Local, TOU, Optimized)

## 📁 Project Structure

```
code/
├── config/                      # YAML configuration files
├── controllers/                 # Battery control algorithms
├── models/                      # Network topology management
├── scenarios/                   # Scenario loading and management
├── simulation/                  # Core simulation engine
├── utils/                       # Cost calculation, stochastic profiles
├── visualization/               # Plotting and network maps
├── venv/data/                   # Network and load profile data
├── main.py                      # Single scenario runner
├── streamlit_app.py            # Interactive web dashboard
├── standalone_optimizer.py     # CVXPY optimization
└── run_stochastic_monte_carlo.py  # Stochastic analysis
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/makarika-microgrid-simulation.git
   cd makarika-microgrid-simulation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Interactive Dashboard
Launch the Streamlit web interface:
```bash
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

#### Single Scenario Simulation
Run a specific scenario:
```bash
python main.py
```
Edit `main.py` to select which scenario to run.

#### Stochastic Analysis
Run Monte Carlo analysis for one scenario:
```bash
python run_stochastic_monte_carlo.py --scenario local_summer_no_marae --n 100
```

Run all 12 scenarios:
```bash
python run_all_stochastic_scenarios.py
```

## 📊 Available Scenarios

The project includes 12 scenarios combining:

| Control Strategy | Season | Event | Scenario ID |
|-----------------|--------|-------|-------------|
| Local | Summer | No Event | `local_summer_no_marae` |
| Local | Summer | Marae Event | `local_summer_marae` |
| Local | Winter | No Event | `local_winter_no_marae` |
| Local | Winter | Marae Event | `local_winter_marae` |
| TOU | Summer | No Event | `tou_summer_no_marae` |
| TOU | Summer | Marae Event | `tou_summer_marae` |
| TOU | Winter | No Event | `tou_winter_no_marae` |
| TOU | Winter | Marae Event | `tou_winter_marae` |
| Optimized | Summer | No Event | `optimized_summer_no_marae` |
| Optimized | Summer | Marae Event | `optimized_summer_marae` |
| Optimized | Winter | No Event | `optimized_winter_no_marae` |
| Optimized | Winter | Marae Event | `optimized_winter_marae` |

## 🏗️ Network Specifications

- **Medium Voltage (MV)**: 11 kV
- **Low Voltage (LV)**: 415 V
- **Transformers**: 2 × MV/LV
- **Loads**: 7 distribution points
- **PV Systems**: 2 solar installations
- **Batteries**: 2 × 20 kWh (5 kW max power)

## 💡 Control Strategies

### Local Control (`SimpleBattery`)
- Autonomous operation based on local voltage
- Charges when voltage > threshold (excess PV)
- Discharges when voltage < threshold (high demand)

### Time-of-Use Control (`TimeOfUseBattery`)
- Responds to electricity price signals
- Charges during off-peak hours (23:00-07:00) @ $0.23/kWh
- Discharges during peak hours (07:00-10:00, 17:00-20:00) @ $0.33/kWh
- Exports during high prices when SOC permits

### Optimized Control (`OptimizedBatteryController`)
- Perfect foresight CVXPY optimization
- Minimizes total daily electricity costs
- Pre-computed schedules from `standalone_optimizer.py`

## 📈 Results

The stochastic analysis generates:
- Cost distributions for each scenario
- Voltage violation statistics
- Battery utilization metrics
- Energy import/export patterns
- Comparative violin plots

## 🔧 Configuration

Edit `config/scenario_definitions.yaml` to modify:
- Battery specifications (capacity, power, efficiency)
- Market prices (peak/off-peak, import/export)
- Controller parameters (SOC limits, charge rates)
- Load profiles and PV profiles

## 📦 Dependencies

Key packages:
- `pandapower` - Power system analysis
- `streamlit` - Interactive dashboard
- `cvxpy` - Optimization
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `plotly` - Interactive plots
- `folium` - Network maps

See `requirements.txt` for complete list.

## 📄 Data Files

Required data files in `venv/data/`:
- `network_waipiro_buses.csv` - Bus definitions
- `network_waipiro_lines.csv` - Line connections
- `network_waipiro_loads.csv` - Load locations
- `network_waipiro_trafos.csv` - Transformers
- `loads_profile.xlsx` - Load time-series data
- `pv_standard_profile.csv` - PV generation profiles



---

**Note**: This project is part of research into renewable energy integration and battery energy storage systems for remote microgrid applications in New Zealand.
