# Entrance Folder - Main Entry Points

This folder contains all standalone entry point scripts for the microgrid simulation framework.

## Available Entry Points

### 1. **main.py** - Single Scenario Simulation
Run a single microgrid scenario with chosen control strategy.

```bash
cd entrance
python main.py
```

Edit `scenario_name` variable in the file to choose different scenarios.

### 2. **streamlit_app.py** - Interactive Web Dashboard
Launch the interactive Streamlit dashboard for visual scenario exploration.

```bash
cd entrance
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

### 3. **run_stochastic_monte_carlo.py** - Single Stochastic Analysis
Run Monte Carlo analysis for ONE scenario with N variations.

```bash
cd entrance
python run_stochastic_monte_carlo.py --scenario local_summer_no_marae --n 100
```

Arguments:
- `--scenario`: Scenario name (e.g., 'local_summer_no_marae')
- `--n`: Number of stochastic variations (default: 10)
- `--output`: Output directory (default: 'results/stochastic')

### 4. **run_all_stochastic_scenarios.py** - Complete Stochastic Analysis
Run Monte Carlo analysis for ALL 12 scenarios (1,200 simulations total).

```bash
cd entrance
python run_all_stochastic_scenarios.py
```

This will:
- Run 100 variations for each of 12 scenario combinations
- Generate violin plots comparing all scenarios
- Save results to `results/stochastic/`

**Warning:** Takes several hours to complete!

## How to Run

All entry points should be run from within the `entrance/` directory:

```powershell
# Activate virtual environment (from project root)
cd "C:\Users\julia\Documents\AA- Master SET\internship\code"
& "venv\Scripts\Activate.ps1"

# Navigate to entrance folder
cd entrance

# Run desired entry point
python main.py
# OR
streamlit run streamlit_app.py
# OR
python run_stochastic_monte_carlo.py --scenario local_summer_no_marae --n 10
# OR
python run_all_stochastic_scenarios.py
```

## Module Access

All scripts in this folder automatically add the parent directory to the Python path, allowing them to import from:
- `models/`
- `scenarios/`
- `simulation/`
- `controllers/`
- `utils/`
- `visualization/`

## Results Location

All simulations save results to (relative to project root):
- `results/{scenario_name}/` - Individual scenario results
- `results/stochastic/` - Stochastic analysis summaries
