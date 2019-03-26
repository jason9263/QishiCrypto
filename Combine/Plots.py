import matplotlib.pyplot as plt

def plot_hist(values=None, bin_count=100, bin_range=None, x_text="X", y_text="Y"):
    plt.hist(values, bins=bin_count, range=bin_range)
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    plt.show()

