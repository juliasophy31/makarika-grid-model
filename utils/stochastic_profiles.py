#!/usr/bin/env python3
"""
Stochastic Profile Generator

Adds Gaussian noise to load and PV profiles.
"""

import numpy as np
import pandas as pd


class StochasticProfileGenerator:
    """Generate stochastic variations of profiles"""
    
    def __init__(
        self,
        load_noise_std: float = 0.10,  # 10% std dev
        pv_noise_std: float = 0.20,    # 20% std dev
        seed: int = None
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


def generate_stochastic_scenario_batch(
    base_load_df: pd.DataFrame,
    base_pv_df: pd.DataFrame,
    pv_column: str,
    n_scenarios: int = 10,
    load_noise_std: float = 0.10,
    pv_noise_std: float = 0.20,
    seed: int = 42
) -> list:
    """
    Generate batch of stochastic scenarios
    
    Returns:
    --------
    scenarios : list of dicts
        Each dict contains:
        - 'id': scenario number (0 to n_scenarios-1)
        - 'load_df': noisy load DataFrame
        - 'pv_df': noisy PV DataFrame
    """
    generator = StochasticProfileGenerator(
        load_noise_std=load_noise_std,
        pv_noise_std=pv_noise_std,
        seed=seed
    )
    
    scenarios = []
    
    for i in range(n_scenarios):
        noisy_load = generator.generate_noisy_load_profile(base_load_df)
        noisy_pv = generator.generate_noisy_pv_profile(base_pv_df, pv_column)
        
        scenarios.append({
            'id': i,
            'load_df': noisy_load,
            'pv_df': noisy_pv
        })
    
    return scenarios


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