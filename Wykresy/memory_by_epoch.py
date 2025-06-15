import matplotlib.pyplot as plt
import numpy as np

# Training data
epochs = [1, 2, 3, 4, 5]

# Memory allocation in GiB for each model
pytorch_memory = [0.958, 0.958, 0.960, 0.959, 0.957]
flux_memory = [8.289, 6.142, 6.142, 6.142, 6.142]
custom_memory = [45.356, 44.497, 44.497, 44.497, 44.497]

# Create the plot
plt.figure(figsize=(12, 7))

# Plot lines with different styles and colors
plt.plot(epochs, pytorch_memory, marker='o', linewidth=2.5, markersize=8, 
         label='PyTorch', color='#FF6B35', linestyle='-')
plt.plot(epochs, flux_memory, marker='s', linewidth=2.5, markersize=8, 
         label='Flux.jl', color='#4ECDC4', linestyle='-')
plt.plot(epochs, custom_memory, marker='^', linewidth=2.5, markersize=8, 
         label='Custom Implementation', color='#9B59B6', linestyle='-')

# Customize the plot
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Memory Allocation (GiB)', fontsize=14, fontweight='bold')
plt.title('Memory Allocation per Epoch Comparison', fontsize=16, fontweight='bold', pad=20)

# Grid and styling
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)

# Set logarithmic scale for Y-axis
plt.yscale('log')

# Set axis limits and ticks
plt.xlim(0.8, 5.2)
plt.ylim(0.5, 200)
plt.xticks(epochs)

# Add horizontal reference lines for better comparison (logarithmic values)
plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Make it look professional
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.8)
plt.gca().spines['bottom'].set_linewidth(0.8)

# Add subtle background color
plt.gca().set_facecolor('#FAFAFA')


# Save the plot
plt.savefig('memory_allocation_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'memory_allocation_comparison.png'")

# Optional: Show the plot (if GUI available)
# plt.show()