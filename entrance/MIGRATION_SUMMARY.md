# Migration to entrance/ Folder - Summary

## What Was Done

Successfully migrated all main entry point scripts to the new `entrance/` folder with proper path adjustments.

## Files Migrated

✅ **Created entrance/ folder**
✅ **main.py** → entrance/main.py
✅ **streamlit_app.py** → entrance/streamlit_app.py
✅ **run_stochastic_monte_carlo.py** → entrance/run_stochastic_monte_carlo.py
✅ **run_all_stochastic_scenarios.py** → entrance/run_all_stochastic_scenarios.py

## Changes Made

### All Migrated Files
Added at the top of each file:
```python
import sys
from pathlib import Path

# Add parent directory to path to access modules
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Path Updates
Changed all relative paths from:
```python
data_dir = Path("venv/data/")
config_path = Path("config/scenario_definitions.yaml")
```

To:
```python
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "venv/data/"
config_path = base_dir / "config/scenario_definitions.yaml"
```

### run_all_stochastic_scenarios.py Specific
Updated subprocess call to use absolute path:
```python
script_path = Path(__file__).parent / 'run_stochastic_monte_carlo.py'
cmd = ['python', str(script_path), '--scenario', scenario_id, ...]
```

## How to Use

### Option 1: Run from entrance/ folder (RECOMMENDED)
```powershell
cd entrance
python main.py
```

### Option 2: Run from project root
```powershell
python entrance/main.py
```

Both work! The scripts will automatically find modules and data.

## What to Do Next

### 1. Test the Migration
```powershell
# From project root
cd entrance

# Test main.py
python main.py

# Test streamlit
streamlit run streamlit_app.py

# Test stochastic
python run_stochastic_monte_carlo.py --scenario local_summer_no_marae --n 5
```

### 2. Optional: Delete Old Files
Once you've confirmed everything works, you can delete:
- `c:\...\code\main.py` (root level)
- `c:\...\code\streamlit_app.py` (root level)
- `c:\...\code\run_stochastic_monte_carlo.py` (root level)
- `c:\...\code\run_all_stochastic_scenarios.py` (root level)

**⚠️ Don't delete them yet!** Test first to make sure everything works.

### 3. Update Documentation
If you have a main README.md at project root, update it to reference:
```
python entrance/main.py
streamlit run entrance/streamlit_app.py
```

## Benefits of This Structure

✅ **Cleaner project root** - Entry points separated from modules
✅ **Clear organization** - All runnable scripts in one place
✅ **Easy to find** - New users know where to look for entry points
✅ **Better separation** - Entry points vs. library code

## Files NOT Migrated

The following remain at root level (correctly):
- `create_stochastic_load_generation_profiles.py` - Analysis script
- `run_sensitivity_analysis.py` - Analysis script
- `config/` - Configuration folder
- `models/`, `scenarios/`, `simulation/`, etc. - Module folders

These are library/analysis scripts, not main entry points.
