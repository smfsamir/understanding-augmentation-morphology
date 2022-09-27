import pdb
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_losses(losses):
    sns.kdeplot(losses)
    plt.tight_layout()
    plt.savefig("results/images/losses.png")
