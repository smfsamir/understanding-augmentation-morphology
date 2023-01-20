import numpy as np
import scipy.special as sp
import pdb
import os
import seaborn as sns
import matplotlib.pyplot as plt
from ..utils.manage_artifacts import savefig, get_artifacts_path

def visualize_losses(losses):
    sns.kdeplot(losses)
    plt.tight_layout()
    plt.savefig("results/images/losses.png")

def visualize_nll_comparison(strategy_nll_frame, all_nll_frame, title):
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    column = 'log(avg. negative log-likelihood)'
    strategy_nll_frame[column] = strategy_nll_frame['nll'].apply(np.log)
    all_nll_frame[column] = all_nll_frame['nll'].apply(np.log)
    sns.histplot(strategy_nll_frame, x=column, hue='strategy', ax=ax1)
    sns.histplot(all_nll_frame, x=column, ax=ax2)
    artifacts_path = get_artifacts_path(f'{title}_nll.png')
    basename = os.path.basename(artifacts_path)
    path = os.path.dirname(artifacts_path)
    ax1.set_title("Subset of augmented data")
    ax2.set_title("All augmented data")
    fig.suptitle(title)
    plt.tight_layout()
    savefig(plt, path, basename)

def visualize_nlls_all(nll_frame):
    nll_frame['log_nll'] = nll_frame['nll'].apply(np.log)
    softmax_arr = sp.log_softmax(nll_frame['log_nll'])
    nll_frame['log_nll_softmax'] = softmax_arr
    _, (ax1, ax2) = plt.subplots(1, 2)
    # sns.displot(nll_frame, x='nll', ax=ax1)
    sns.displot(nll_frame, x='log_nll_softmax', ax=ax2)
    artifacts_path = get_artifacts_path('nll_softmax_log.png')
    basename = os.path.basename(artifacts_path)
    path = os.path.dirname(artifacts_path)
    savefig(plt, path, basename)
    plt.tight_layout()

def visualize_uat_selection(selection_frame, use_empirical):
    sns.countplot(data=selection_frame, x='tag', hue='strategy')
    artifacts_path = get_artifacts_path(f'uat_tag_dist_use_empirical={use_empirical}.png')
    basename = os.path.basename(artifacts_path)
    path = os.path.dirname(artifacts_path)
    plt.tight_layout()
    savefig(plt, path, basename)
