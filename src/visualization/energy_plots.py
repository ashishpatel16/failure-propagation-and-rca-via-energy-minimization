import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_academic_plot_params():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

def plot_energy_vs_probability(energies):
    """
    Plots Probability P(L) vs Energy E(L).
    """
    setup_academic_plot_params()
    
    energies_arr = np.array(energies)
    
    # Filter out the 'infinity' fallbacks (1e9) to keep the plot simple
    valid_energies = energies_arr[energies_arr < 1e8]
    if len(valid_energies) == 0:
        return
        
    # Shift energies to prevent numerical issues
    min_e = np.min(valid_energies)
    shifted = valid_energies - min_e
    
    # Compute Gibbs probabilities
    probs = np.exp(-shifted)
    probs = probs / np.sum(probs)
    
    plt.figure(figsize=(8, 5))
    
    if len(valid_energies) > 1:
        ax2 = plt.gca().twinx()
        sns.kdeplot(
            x=valid_energies, 
            fill=True, 
            color="#3498db", 
            alpha=0.2,
            ax=ax2,
            linewidth=1.5
        )
        ax2.set_ylabel("Density of States (KDE)", fontsize=11, color="#2980b9")
        ax2.tick_params(axis='y', colors='#2980b9')
        ax2.grid(False)  
        
    plt.scatter(valid_energies, probs, color='#2c3e50', alpha=0.8, zorder=3)
    
    plt.title("Gibbs Distribution: Probability vs Energy")
    plt.xlabel("Energy E(L)")
    plt.ylabel("Probability P(L)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_energy_vs_probability_lambdas(lambdas, energy_map, temperature: float):
    """
    Computes energies for different lambda values and plots Probability P(L) vs Energy E(L)
    for each lambda on the same figure.
    
    Args:
        lambdas: List of lambda values.
        energy_map: Dict mapping lambda to a list of energies for all labelings.
        temperature: The temperature for the Gibbs distribution.
    """
    setup_academic_plot_params()
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e']
    
    for i, lambd in enumerate(lambdas):
        energies = energy_map[lambd]
        energies_arr = np.array(energies)
        valid_energies = energies_arr[energies_arr < 1e8]
        if len(valid_energies) == 0:
            continue
            
        min_e = np.min(valid_energies)
        shifted = valid_energies - min_e
        probs = np.exp(-shifted / temperature)
        probs = probs / np.sum(probs)
        
        color = colors[i % len(colors)]
        plt.scatter(valid_energies, probs, color=color, alpha=0.6, label=f"λ={lambd}", zorder=3)
        
    plt.title(f"Gibbs Distribution: Probability vs Energy (T={temperature})")
    plt.xlabel("Energy E(L)")
    plt.ylabel("Probability P(L)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
