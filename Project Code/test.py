import matplotlib.pyplot as plt
import numpy as np

# Data for the 4 groups
labels = ['1D-CNN Only', 'BiGRU Only', 'Proposed Framework']
version_A = [96.11, 93.48, 97.92]
version_B = [94.14, 92.78, 96.61]

# Positions for the bars
spacing = 2.5  # increase this to create more space between groups
x = np.arange(len(labels)) * spacing
width = 0.65  

# Create the plot
fig, ax = plt.subplots()

ax.bar(x - width/2, version_A, width, label='UCI_2015')
ax.bar(x + width/2, version_B, width, label='Mendeley_2020')

# Labels and title
ax.set_xlabel('Individual Components')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Individual Architectural Components on each dataset.')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.set_ylim(90, 100)

plt.savefig("comparison_bar_graph.png", dpi=300, bbox_inches="tight")
plt.show()