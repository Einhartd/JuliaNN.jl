import matplotlib.pyplot as plt
import numpy as np

# Training data
epochs = [1, 2, 3, 4, 5]

# Loss values for each model
pytorch_loss = [0.4680, 0.2993, 0.2395, 0.1931, 0.1543]
flux_loss = [0.5721, 0.3357, 0.2488, 0.1905, 0.1400]
custom_loss = [0.6205, 0.3737, 0.2709, 0.2009, 0.1445]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines with different styles and colors
plt.plot(epochs, pytorch_loss, marker='o', linewidth=2.5, markersize=8, 
         label='PyTorch', color='#FF6B35', linestyle='-')
plt.plot(epochs, flux_loss, marker='s', linewidth=2.5, markersize=8, 
         label='Flux.jl', color='#4ECDC4', linestyle='-')
plt.plot(epochs, custom_loss, marker='^', linewidth=2.5, markersize=8, 
         label='Custom Implementation', color='#9B59B6', linestyle='-')

# Customize the plot
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Training Loss', fontsize=14, fontweight='bold')
plt.title('Training Loss Comparison Across Epochs', fontsize=16, fontweight='bold', pad=20)

# Grid and styling
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)

# Set axis limits and ticks
plt.xlim(0.8, 5.2)
plt.ylim(0.1, 0.65)
plt.xticks(epochs)
plt.yticks(np.arange(0.1, 0.7, 0.1))

# Make it look professional
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.8)
plt.gca().spines['bottom'].set_linewidth(0.8)

# Add subtle background color
plt.gca().set_facecolor('#FAFAFA')

# Save the plot instead of showing it
plt.savefig('training_loss_comparison.png', dpi=600, bbox_inches='tight')
print("Plot saved as 'training_loss_comparison.png'")

# Optional: Show the plot (if GUI available)
# plt.show()