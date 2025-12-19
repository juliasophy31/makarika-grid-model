"""
Stochastic Profile Generator

Adds Gaussian noise to load and PV profiles for Monte Carlo analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class StochasticProfileGenerator:
    """Generate stochastic variations of profiles"""
    
    def __init__(
        self,
        load_noise_std: float = 0.10,  # 10% std dev
        pv_noise_std: float = 0.20,    # 20% std dev
        seed: Optional[int] = None
    ):
        self.load_noise_std = load_noise_std
        self.pv_noise_std = pv_noise_std
        self.rng = np.random.RandomState(seed)
    
    def generate_noisy_load_profile(
        self,
        base_load_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add Gaussian noise to load profile
        
        Parameters:
        -----------
        base_load_df : pd.DataFrame
            Columns = load names (e.g., 'load_bus10_kw')
            Rows = timesteps (0-23)
        
        Returns:
        --------
        noisy_load_df : pd.DataFrame
            Same structure with noise added
        """
        noisy_df = base_load_df.copy()
        
        for col in noisy_df.columns:
            # Generate noise multiplier: mean=1.0, std=load_noise_std
            noise = self.rng.normal(1.0, self.load_noise_std, size=len(noisy_df))
            
            # Apply noise
            noisy_df[col] = noisy_df[col] * noise
            
            # Ensure non-negative
            noisy_df[col] = noisy_df[col].clip(lower=0)
        
        return noisy_df
    
    def generate_noisy_pv_profile(
        self,
        base_pv_df: pd.DataFrame,
        capacity_factor_column: str
    ) -> pd.DataFrame:
        """
        Add Gaussian noise to PV capacity factors
        
        Parameters:
        -----------
        base_pv_df : pd.DataFrame
            Must contain capacity_factor_column
        capacity_factor_column : str
            e.g., 'capacity_factor_winter'
        
        Returns:
        --------
        noisy_pv_df : pd.DataFrame
            Same structure with noisy capacity factors
        """
        noisy_df = base_pv_df.copy()
        
        # Generate noise
        cf_values = noisy_df[capacity_factor_column].values
        noise = self.rng.normal(1.0, self.pv_noise_std, size=len(cf_values))
        
        # Apply noise
        noisy_cf = cf_values * noise
        
        # Clip to valid range [0, 1]
        noisy_cf = np.clip(noisy_cf, 0.0, 1.0)
        
        # Update column
        noisy_df[capacity_factor_column] = noisy_cf
        
        return noisy_df


class StochasticScenarioManager:
    """Manages multiple stochastic runs of scenarios"""
    
    def __init__(
        self,
        scenario_manager,
        n_runs: int = 10,
        load_noise_std: float = 0.10,
        pv_noise_std: float = 0.20,
        seed: int = 42
    ):
        self.scenario_manager = scenario_manager
        self.n_runs = n_runs
        self.load_noise_std = load_noise_std
        self.pv_noise_std = pv_noise_std
        self.seed = seed
        self.generator = StochasticProfileGenerator(
            load_noise_std=load_noise_std,
            pv_noise_std=pv_noise_std,
            seed=seed
        )
    
    def generate_stochastic_profiles(
        self,
        scenario_name: str,
        run_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate stochastic profiles for a specific scenario and run
        
        Parameters:
        -----------
        scenario_name : str
            Base scenario to modify
        run_id : int
            Unique run identifier (affects random seed)
        
        Returns:
        --------
        noisy_load_df : pd.DataFrame
            Load profile with noise
        noisy_pv_df : pd.DataFrame  
            PV profile with noise
        """
        # Set run-specific seed
        run_seed = self.seed + run_id if self.seed else None
        self.generator.rng = np.random.RandomState(run_seed)
        
        # Get base scenario configuration
        if scenario_name not in self.scenario_manager.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenario_manager.scenarios[scenario_name]
        
        # Load base profiles
        base_load_df, base_pv_df = self._load_base_profiles(scenario)
        
        # Generate noisy profiles
        noisy_load_df = self.generator.generate_noisy_load_profile(base_load_df)
        
        # For PV, we need to determine the capacity factor column
        pv_config = scenario.get('pv_systems', {})
        capacity_factor_column = pv_config.get('profile_column', 'capacity_factor')
        noisy_pv_df = self.generator.generate_noisy_pv_profile(base_pv_df, capacity_factor_column)
        
        return noisy_load_df, noisy_pv_df
    
    def _load_base_profiles(self, scenario) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load base profiles for a scenario"""
        # Load load profile
        load_config = scenario.get('loads', {})
        profile_file = self.scenario_manager.data_dir / 'loads_profile.xlsx'
        
        if profile_file.exists():
            sheet_name = load_config.get('sheet_name', 'winter_no_marae')
            load_df = pd.read_excel(profile_file, sheet_name=sheet_name, index_col=0)
        else:
            raise FileNotFoundError(f"Load profile not found at {profile_file}")
        
        # Load PV profile
        pv_config = scenario.get('pv_systems', {})
        pv_file = self.scenario_manager.data_dir / 'pv_standard_profile.csv'
        delimiter = pv_config.get('delimiter', ';')
        
        if pv_file.exists():
            pv_df = pd.read_csv(pv_file, delimiter=delimiter, index_col=0)
        else:
            raise FileNotFoundError(f"PV profile not found at {pv_file}")
        
        return load_df, pv_df
    
    def apply_stochastic_scenario(
        self,
        scenario_name: str,
        run_id: int
    ):
        """
        Apply a stochastic version of a scenario using custom profiles
        
        Parameters:
        -----------
        scenario_name : str
            Base scenario name
        run_id : int
            Stochastic run ID
        """
        # Generate stochastic profiles
        noisy_load_df, noisy_pv_df = self.generate_stochastic_profiles(scenario_name, run_id)
        
        # Use the new clean interface with custom profiles
        result = self.scenario_manager.apply_scenario_with_custom_profiles(
            scenario_name,
            load_profiles=noisy_load_df,
            pv_profiles=noisy_pv_df
        )
        
        print(f"📊 Applied stochastic scenario {scenario_name} (run {run_id})")
        return result


# Test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/mnt/project')
    
    # Load base profiles
    load_df = pd.read_excel(
        'loads_profile.xlsx',
        sheet_name='winter_no_marae',
        index_col=0
    )
    
    pv_df = pd.read_csv(
        'pv_standard_profile.csv',
        delimiter=';',
        index_col=0
    )
    
    # Generate scenarios
    scenarios = generate_stochastic_scenario_batch(
        base_load_df=load_df,
        base_pv_df=pv_df,
        pv_column='capacity_factor_winter',
        n_scenarios=10
    )
    
    print(f"✅ Generated {len(scenarios)} scenarios")
    
    # Show statistics
    base_total = load_df.sum().sum()
    stochastic_totals = [s['load_df'].sum().sum() for s in scenarios]
    
    print(f"\nLoad Total Statistics:")
    print(f"  Base:  {base_total:.1f} kWh")
    print(f"  Mean:  {np.mean(stochastic_totals):.1f} kWh")
    print(f"  Std:   {np.std(stochastic_totals):.1f} kWh")
    print(f"  CV:    {np.std(stochastic_totals)/np.mean(stochastic_totals)*100:.1f}%")