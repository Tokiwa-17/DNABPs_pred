import matplotlib.pyplot as plt
import numpy as np

idx = [1, 2, 3, 4, 5]
acc = [71.50, 74.34, 76.34, 74.73, 72.58]
x = np.linspace(0, 2, 100)  # Sample data.

# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(idx, acc, label='acc')  # Plot some data on the axes.
ax.set_xlabel('d')  # Add an x-label to the axes.
ax.set_ylabel('accuracy')  # Add a y-label to the axes.
ax.set_title("accuracy changes in pairwise distance")  # Add a title to the axes.
ax.legend()  # A
plt.show()