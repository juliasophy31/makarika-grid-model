"""
Run Stochastic Monte Carlo for All Scenarios and Create Violin Plots
=====================================================================
Runs 100 stochastic scenarios for each condition and creates violin plots
showing cost distributions.
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run_stochastic_for_scenario(controller, season, marae, n_scenarios=100):
    """Run stochastic Monte Carlo for a specific scenario."""
    
    scenario_id = f"{controller}_{season}_{marae}"
    
    print(f"\n{'='*80}")
    print(f"Running Stochastic Analysis: {scenario_id}")
    print(f"{'='*80}")
    
    # Run the stochastic Monte Carlo script
    cmd = [
        'python',
        'run_stochastic_monte_carlo.py',
        '--scenario', scenario_id,
        '--n', str(n_scenarios),
        '--output', 'results/stochastic'
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ Completed: {scenario_id}")
    else:
        print(f"❌ Failed: {scenario_id}")
    
    return result.returncode == 0


def load_stochastic_results(controller, season, marae):
    """Load stochastic results for a scenario."""
    
    scenario_id = f"{controller}_{season}_{marae}"
    summary_file = Path(f"results/stochastic/{controller}_{season}_{marae}_stochastic_summary.csv")
    
    if not summary_file.exists():
        print(f"⚠️  Warning: {summary_file} not found")
        return None
    
    df = pd.read_csv(summary_file)
    return df


def create_cost_violin_plots():
    """Create violin plots for cost distributions across all scenarios."""
    
    # Define scenarios
    scenarios = [
        {'season': 'summer', 'marae': 'no_marae', 'title': 'Summer - No Marae'},
        {'season': 'summer', 'marae': 'marae', 'title': 'Summer - Marae Event'},
        {'season': 'winter', 'marae': 'no_marae', 'title': 'Winter - No Marae'},
        {'season': 'winter', 'marae': 'marae', 'title': 'Winter - Marae Event'}
    ]
    
    controllers = ['local', 'tou', 'optimized']
    
    controller_labels = {
        'local': 'Local Control',
        'tou': 'TOU Control',
        'optimized': 'Optimized Control'
    }
    
    colors = {
        'local': '#d62728',      # Red
        'tou': '#1f77b4',        # Blue
        'optimized': '#2ca02c'   # Green
    }
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Daily Cost Distribution - Stochastic Analysis (100 Scenarios)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, scenario_info in enumerate(scenarios):
        ax = axes[idx]
        season = scenario_info['season']
        marae = scenario_info['marae']
        title = scenario_info['title']
        
        # Collect data for this scenario
        violin_data = []
        labels = []
        colors_list = []
        
        for controller in controllers:
            df = load_stochastic_results(controller, season, marae)
            
            if df is not None and 'total_cost' in df.columns:
                # Remove NaN values
                costs = df['total_cost'].dropna()
                
                if len(costs) > 0:
                    violin_data.append(costs)
                    labels.append(controller_labels[controller])
                    colors_list.append(colors[controller])
                    
                    # Print statistics
                    print(f"\n{scenario_info['title']} - {controller_labels[controller]}:")
                    print(f"  Mean: ${costs.mean():.2f}")
                    print(f"  Std:  ${costs.std():.2f}")
                    print(f"  Min:  ${costs.min():.2f}")
                    print(f"  Max:  ${costs.max():.2f}")
        
        if violin_data:
            # Create violin plot
            parts = ax.violinplot(violin_data, positions=range(len(violin_data)),
                                 showmeans=True, showmedians=True, widths=0.7)
            
            # Color the violins
            for pc, color in zip(parts['bodies'], colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Style the auxiliary lines
            for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1.5)
            
            # Set labels
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
            ax.set_ylabel('Daily Cost ($)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add text with statistics
            for i, (data, label, color) in enumerate(zip(violin_data, labels, colors_list)):
                mean_val = data.mean()
                ax.text(i, mean_val, f'${mean_val:.1f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='gray')
            ax.set_ylabel('Daily Cost ($)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    # Save
    output_dir = Path('results/stochastic')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'cost_violin_plots_all_scenarios.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")
    
    plt.close()


if __name__ == '__main__':
    
    # Define all scenarios to run
    scenarios_to_run = [
        # ('local', 'summer', 'no_marae'),  # Already done
        ('local', 'summer', 'marae'),
        ('local', 'winter', 'no_marae'),
        ('local', 'winter', 'marae'),
        ('tou', 'summer', 'no_marae'),
        ('tou', 'summer', 'marae'),
        ('tou', 'winter', 'no_marae'),
        ('tou', 'winter', 'marae'),
        ('optimized', 'summer', 'no_marae'),
        ('optimized', 'summer', 'marae'),
        ('optimized', 'winter', 'no_marae'),
        ('optimized', 'winter', 'marae')
    ]
    
    print("=" * 80)
    print("RUNNING STOCHASTIC MONTE CARLO FOR ALL SCENARIOS")
    print("=" * 80)
    print(f"\nTotal scenarios to run: {len(scenarios_to_run)}")
    print("Each scenario: 100 stochastic variations")
    print("This may take a while...\n")
    
    # Run all scenarios
    success_count = 0
    for controller, season, marae in scenarios_to_run:
        success = run_stochastic_for_scenario(controller, season, marae, n_scenarios=100)
        if success:
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {success_count}/{len(scenarios_to_run)} scenarios")
    print(f"{'='*80}")
    
    # Create violin plots
    print("\n" + "=" * 80)
    print("CREATING VIOLIN PLOTS")
    print("=" * 80)
    
    create_cost_violin_plots()
    
    print("\n" + "=" * 80)
    print("✓ All tasks completed!")
    print("=" * 80)
