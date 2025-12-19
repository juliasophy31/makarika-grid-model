"""
Create load and generation profiles with stochastic noise for all scenarios.
Similar to the base load/generation plot but with stochastic variations shown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.stochastic_profiles import StochasticProfileGenerator


def create_stochastic_profile_plot(n_scenarios=20, alpha=0.15):
    """
    Create a plot showing load and generation profiles with stochastic noise
    for all scenarios (winter/summer, marae/no marae).
    
    Parameters:
    -----------
    n_scenarios : int
        Number of stochastic variations to generate per scenario
    alpha : float
        Transparency of stochastic lines (0-1)
    """
    
    # Initialize profile generator
    profile_gen = StochasticProfileGenerator(
        load_noise_std=0.10,  # 10% std dev for load
        pv_noise_std=0.20,    # 20% std dev for PV
        seed=42
    )
    
    # Load base profiles
    load_excel = pd.read_excel('venv/data/loads_profile.xlsx', sheet_name=None, index_col=0)
    pv_df = pd.read_csv('venv/data/pv_standard_profile.csv', sep=';', index_col=0)
    
    # Total PV capacity: 2 systems * 18 kW = 36 kW
    pv_capacity = 36.0
    
    # Define scenarios
    scenarios = {
        'winter_no_marae': {'load_sheet': 'winter_no_marae', 'pv_col': 'capacity_factor_winter', 'season': 'Winter'},
        'winter_marae': {'load_sheet': 'winter_marae', 'pv_col': 'capacity_factor_winter', 'season': 'Winter'},
        'summer_no_marae': {'load_sheet': 'summer_no_marae', 'pv_col': 'capacity_factor_summer', 'season': 'Summer'},
        'summer_marae': {'load_sheet': 'summer_marae', 'pv_col': 'capacity_factor_summer', 'season': 'Summer'}
    }
    
    # Define colors - lighter for stochastic, darker for mean
    colors = {
        'winter_no_marae': '#4169E1',      # Royal blue (mean)
        'winter_marae': '#8A2BE2',         # Blue violet (mean)
        'summer_no_marae': '#32CD32',      # Lime green (mean)
        'summer_marae': '#228B22',         # Forest green (mean)
        'winter_gen': '#FFA500',           # Orange (dashed, mean)
        'summer_gen': '#FFD700',           # Gold (dashed, mean)
    }
    
    # Lighter colors for stochastic variations
    stochastic_colors = {
        'winter_no_marae': '#B0C4DE',      # Light steel blue
        'winter_marae': '#DDA0DD',         # Plum
        'summer_no_marae': '#90EE90',      # Light green
        'summer_marae': '#8FBC8F',         # Dark sea green
        'winter_gen': '#FFD7A3',           # Light orange/peach
        'summer_gen': '#FFF8B3',           # Light yellow
    }
    
    hours = range(24)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Store base profiles for legend ordering
    base_profiles = {}
    
    # Process each scenario
    for scenario_id, config in scenarios.items():
        load_df = load_excel[config['load_sheet']]
        pv_column = config['pv_col']
        season = config['season']
        
        # Calculate base profiles
        base_load = load_df.sum(axis=1)
        base_pv = pv_df[pv_column] * pv_capacity
        
        base_profiles[scenario_id] = {
            'load': base_load,
            'pv': base_pv,
            'config': config
        }
        
        # Generate and plot stochastic load variations
        for i in range(n_scenarios):
            noisy_load_df = profile_gen.generate_noisy_load_profile(load_df)
            agg_load = noisy_load_df.sum(axis=1)
            
            # Only add label for first iteration
            label = None
            ax.plot(hours, agg_load, color=stochastic_colors[scenario_id], 
                   alpha=0.4, linewidth=0.6, label=label)
    
    # Generate and plot stochastic PV variations
    for season_name, pv_col, gen_color in [('Winter', 'capacity_factor_winter', 'winter_gen'), 
                                            ('Summer', 'capacity_factor_summer', 'summer_gen')]:
        base_pv = pv_df[pv_col] * pv_capacity
        
        for i in range(n_scenarios):
            # Create temporary dataframe for PV profile
            temp_pv_df = pd.DataFrame({pv_col: pv_df[pv_col]})
            noisy_pv_df = profile_gen.generate_noisy_pv_profile(temp_pv_df, pv_col)
            agg_pv = noisy_pv_df[pv_col] * pv_capacity
            
            ax.plot(hours, agg_pv, color=stochastic_colors[gen_color], 
                   alpha=0.4, linewidth=0.6, linestyle='--')
    
    # Plot base (mean) profiles on top with thicker lines
    # Generation first (behind loads)
    for season_name, pv_col, gen_color in [('Winter', 'capacity_factor_winter', 'winter_gen'), 
                                            ('Summer', 'capacity_factor_summer', 'summer_gen')]:
        base_pv = pv_df[pv_col] * pv_capacity
        ax.plot(hours, base_pv, color=colors[gen_color], linewidth=3.0, 
               linestyle='--', label=f'{season_name} - Generation', zorder=10)
    
    # Then loads
    for scenario_id, data in base_profiles.items():
        config = data['config']
        season = config['season']
        marae_status = 'Marae Event - Load' if 'marae' in scenario_id and 'no_marae' not in scenario_id else 'No Marae - Load'
        label = f'{season} - {marae_status}'
        
        ax.plot(hours, data['load'], color=colors[scenario_id], 
               linewidth=3.0, label=label, zorder=10)
    
    # Formatting
    ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power (kW)', fontsize=14, fontweight='bold')
    ax.set_title('Load and Generation Profiles with Stochastic Variations\n' + 
                f'({n_scenarios} Monte Carlo scenarios per profile)',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 65)
    
    # Set x-axis ticks
    ax.set_xticks(range(0, 24, 2))
    
    # Legend - order to match original image
    # Generation (dashed) then Loads (solid)
    handles, labels = ax.get_legend_handles_labels()
    
    # Reorder: Winter Gen, Summer Gen, then loads in order
    order = []
    label_order = [
        'Winter - Generation',
        'Summer - Generation', 
        'Winter - No Marae - Load',
        'Winter - Marae Event - Load',
        'Summer - No Marae - Load',
        'Summer - Marae Event - Load'
    ]
    
    for desired_label in label_order:
        for i, label in enumerate(labels):
            if label == desired_label:
                order.append(i)
                break
    
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
             fontsize=10, loc='upper left', framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('results/overview/load_generation_profiles_stochastic.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


if __name__ == '__main__':
    print("=" * 80)
    print("CREATING STOCHASTIC LOAD & GENERATION PROFILE PLOT")
    print("=" * 80)
    
    fig = create_stochastic_profile_plot(n_scenarios=100, alpha=0.15)
    
    print("\n" + "=" * 80)
    print("✓ Plot generation complete!")
    print("=" * 80)
