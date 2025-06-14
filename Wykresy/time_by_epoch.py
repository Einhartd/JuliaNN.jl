import matplotlib.pyplot as plt
import numpy as np

# Training data
epochs = [1, 2, 3, 4, 5]

# Epoch times in seconds for each model
pytorch_times = [9.3494, 10.3062, 10.2606, 10.0953, 9.9673]
flux_times = [52.3239, 12.8267, 13.3807, 13.8692, 13.8960]
custom_times = [48.3290, 37.5116, 37.3129, 37.2039, 37.8762]

# Create the plot
plt.figure(figsize=(12, 7))

# Plot lines with different styles and colors
plt.plot(epochs, pytorch_times, marker='o', linewidth=2.5, markersize=8, 
         label='PyTorch', color='#FF6B35', linestyle='-')
plt.plot(epochs, flux_times, marker='s', linewidth=2.5, markersize=8, 
         label='Flux.jl', color='#4ECDC4', linestyle='-')
plt.plot(epochs, custom_times, marker='^', linewidth=2.5, markersize=8, 
         label='Custom Implementation', color='#9B59B6', linestyle='-')

# Customize the plot
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
plt.title('Training Time per Epoch Comparison', fontsize=16, fontweight='bold', pad=20)

# Grid and styling
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)

# Set axis limits and ticks
plt.xlim(0.8, 5.2)
plt.ylim(0, 55)
plt.xticks(epochs)

# Add horizontal reference lines for better comparison
plt.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=20, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=30, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=40, color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Make it look professional
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.8)
plt.gca().spines['bottom'].set_linewidth(0.8)

# Add subtle background color
plt.gca().set_facecolor('#FAFAFA')

# Add annotations for interesting points
plt.annotate('First epoch overhead', 
             xy=(1, 52.3239), xytext=(1.25, 53),
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             fontsize=10, color='gray')

plt.annotate('First epoch overhead', 
             xy=(1, 48.3290), xytext=(1.25, 47),
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             fontsize=10, color='gray')

# Save the plot instead of showing it
plt.savefig('epoch_time_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'epoch_time_comparison.png'")

# Optional: Show the plot (if GUI available)
# plt.show()