from model import AAVELendingModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter  # For curve smoothing

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_simulation_with_sensitivity(reputation_sensitivity, steps=400, seed=42):
    """Run a simulation with a specific reputation sensitivity value."""
    print(f"\n=== Running simulation with sensitivity: {reputation_sensitivity} ===\n")
    
    model = AAVELendingModel(
        num_lenders=20,
        num_borrowers=50,
        base_collateral_factor=0.75,
        reputation_sensitivity=reputation_sensitivity,
        liquidation_threshold=1.05,
        base_interest_rate=0.03,
        seed=seed  # Use the same seed for all simulations
    )
    
    # Track metrics over time
    metrics = {
        'utilization_rate': [],
        'liquidations': [],
        'avg_reputation': [],
        'avg_collateral_ratio': []
    }
    
    # Run the simulation
    for i in range(steps):
        model.step()
        
        # Record metrics
        metrics['utilization_rate'].append(model.get_utilization_rate())
        metrics['liquidations'].append(model.total_liquidations)
        metrics['avg_reputation'].append(model.get_average_reputation())
        metrics['avg_collateral_ratio'].append(model.get_average_collateral_ratio())
    
    # Print final stats
    print(f"Final metrics for sensitivity {reputation_sensitivity}:")
    print(f"  Total Liquidity: {model.total_liquidity:.2f}")
    print(f"  Total Borrowed: {model.total_borrowed:.2f}")
    print(f"  Utilization Rate: {model.get_utilization_rate():.2f}")
    print(f"  Total Liquidations: {model.total_liquidations}")
    print(f"  Average Reputation: {model.get_average_reputation():.2f}")
    print(f"  Average Collateral Ratio: {model.get_average_collateral_ratio():.2f}")
    
    # Count liquidated borrowers
    liquidated = sum(1 for agent in model.schedule.agents 
                    if hasattr(agent, 'is_liquidated') and agent.is_liquidated)
    print(f"  Liquidated Borrowers: {liquidated}")
    
    return metrics, model

def compare_sensitivities():
    """Run simulations with different sensitivity values and compare results."""
    sensitivity_values = [0.0, 0.2, 0.4, 0.6]
    results = {}
    final_metrics = []
    
    # Use the same seed for all simulations to ensure consistency
    fixed_seed = 42
    
    for sensitivity in sensitivity_values:
        metrics, model = run_simulation_with_sensitivity(sensitivity, seed=fixed_seed)
        results[sensitivity] = metrics
        
        # Record final metrics for comparison
        final_metrics.append({
            'Sensitivity': sensitivity,
            'Liquidations': model.total_liquidations,
            'Utilization_Rate': model.get_utilization_rate(),
            'Avg_Reputation': model.get_average_reputation(),
            'Avg_Collateral_Ratio': model.get_average_collateral_ratio()
        })
    
    # Create a DataFrame for final metrics
    df = pd.DataFrame(final_metrics)
    print("\nComparison of final metrics:")
    print(df)
    
    # Plot comparison charts
    plot_comparison(results, sensitivity_values)
    
    return results, df

def plot_comparison(results, sensitivity_values):
    """Plot comparison charts for different sensitivity values."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Utilization Rates
    for sensitivity in sensitivity_values:
        data = results[sensitivity]['utilization_rate']
        # Apply curve smoothing if enough data points
        if len(data) > 10:
            window_size = min(15, len(data) - (1 if len(data) % 2 == 0 else 0))
            if window_size > 2:
                smoothed_data = savgol_filter(data, window_size, 2)
                axes[0, 0].plot(smoothed_data, 
                          label=f'Sensitivity={sensitivity}', 
                          linewidth=2.5)
                # Plot original data as semi-transparent
                axes[0, 0].plot(data, alpha=0.2, linestyle=':', 
                          label='_nolegend_')
            else:
                axes[0, 0].plot(data, label=f'Sensitivity={sensitivity}')
        else:
            axes[0, 0].plot(data, label=f'Sensitivity={sensitivity}')
    
    axes[0, 0].set_title('Utilization Rate Over Time')
    axes[0, 0].set_ylabel('Utilization Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot Liquidations
    for sensitivity in sensitivity_values:
        data = results[sensitivity]['liquidations']
        # Apply curve smoothing if enough data points
        if len(data) > 10:
            window_size = min(15, len(data) - (1 if len(data) % 2 == 0 else 0))
            if window_size > 2:
                smoothed_data = savgol_filter(data, window_size, 2)
                axes[0, 1].plot(smoothed_data, 
                          label=f'Sensitivity={sensitivity}', 
                          linewidth=2.5)
                # Plot original data as semi-transparent
                axes[0, 1].plot(data, alpha=0.2, linestyle=':', 
                          label='_nolegend_')
            else:
                axes[0, 1].plot(data, label=f'Sensitivity={sensitivity}')
        else:
            axes[0, 1].plot(data, label=f'Sensitivity={sensitivity}')
    
    axes[0, 1].set_title('Cumulative Liquidations Over Time')
    axes[0, 1].set_ylabel('Liquidations')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Average Reputation
    for sensitivity in sensitivity_values:
        data = results[sensitivity]['avg_reputation']
        # Apply curve smoothing if enough data points
        if len(data) > 10:
            window_size = min(15, len(data) - (1 if len(data) % 2 == 0 else 0))
            if window_size > 2:
                smoothed_data = savgol_filter(data, window_size, 2)
                axes[1, 0].plot(smoothed_data, 
                          label=f'Sensitivity={sensitivity}', 
                          linewidth=2.5)
                # Plot original data as semi-transparent
                axes[1, 0].plot(data, alpha=0.2, linestyle=':', 
                          label='_nolegend_')
            else:
                axes[1, 0].plot(data, label=f'Sensitivity={sensitivity}')
        else:
            axes[1, 0].plot(data, label=f'Sensitivity={sensitivity}')
    
    axes[1, 0].set_title('Average Reputation Over Time')
    axes[1, 0].set_ylabel('Avg Reputation')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot Average Collateral Ratio
    for sensitivity in sensitivity_values:
        data = results[sensitivity]['avg_collateral_ratio']
        # Apply curve smoothing if enough data points
        if len(data) > 10:
            window_size = min(15, len(data) - (1 if len(data) % 2 == 0 else 0))
            if window_size > 2:
                smoothed_data = savgol_filter(data, window_size, 2)
                axes[1, 1].plot(smoothed_data, 
                          label=f'Sensitivity={sensitivity}', 
                          linewidth=2.5)
                # Plot original data as semi-transparent
                axes[1, 1].plot(data, alpha=0.2, linestyle=':', 
                          label='_nolegend_')
            else:
                axes[1, 1].plot(data, label=f'Sensitivity={sensitivity}')
        else:
            axes[1, 1].plot(data, label=f'Sensitivity={sensitivity}')
    
    axes[1, 1].set_title('Average Collateral Ratio Over Time')
    axes[1, 1].set_ylabel('Avg Collateral Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('sensitivity_comparison.png')
    plt.show()

if __name__ == "__main__":
    results, df = compare_sensitivities()
    
    # Save results to CSV
    df.to_csv('sensitivity_comparison_results.csv', index=False) 