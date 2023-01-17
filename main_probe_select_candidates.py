import pdb
import sys
from typing import Dict
import pickle as pkl


import numpy as np
import pandas as pd

from packages.fairseq_utils.dataloading_utils import get_initial_generation_frame, get_augmentation_example_lengths
from packages.utils.util_functions import get_number_test_examples, get_initial_model_path, generate_hyperparams
from packages.augmentation.subset_selecter_strategy import get_subset_selecter 
from packages.utils.constants import LANGUAGES
from packages.visualizations.visualize import visualize_nll_comparison, visualize_nlls_all, visualize_uat_selection

def load_augment_likelihoods(language: str, **kwargs) -> pd.DataFrame:
    initial_path = get_initial_model_path(language, **kwargs)
    likelihood_pkl_path = f"{initial_path}/{language}_log_likelihoods.pickle"
    with open(likelihood_pkl_path, 'rb') as likelihoods_pkl:
        example_likelihoods = pkl.load(likelihoods_pkl)
    indices, log_likelihoods = zip(*example_likelihoods)
    negative_log_likelihoods = -np.array(log_likelihoods)
    likelihood_frame = pd.DataFrame({
        "nll": negative_log_likelihoods,
    }, index=indices)
    return likelihood_frame

def inspect_augmentation_candidates(language: str, augmentation_type: str, \
									hyperparams: Dict) -> pd.DataFrame:
    likelihood_frame = load_augment_likelihoods(language, **hyperparams)
    initial_generation_frame = get_initial_generation_frame(language) # contains gold test + original 10,000 test examples.
    num_gold_test_examples = get_number_test_examples(language)
    augment_example_lengths = get_augmentation_example_lengths(initial_generation_frame, num_gold_test_examples)
    subset_sampler = get_subset_selecter(language, augmentation_type, get_initial_model_path(language, **hyperparams), initial_generation_frame, num_gold_test_examples, augment_example_lengths, **hyperparams)
    # TODO: need to prefix with the number of points that are selected.
    subset_augmentation_frame = subset_sampler.get_best_points(hyperparams['num_aug'])
    subset_augmentation_frame = subset_augmentation_frame.reset_index()
    subset_augmentation_frame = subset_augmentation_frame.merge(likelihood_frame, left_on='index', right_index=True)
    print(subset_augmentation_frame['tgt'].value_counts())
    return subset_augmentation_frame

def visualize_uncertainty(language):
    hyperparams = {"train_medium": True}
    likelihood_frame = load_augment_likelihoods(language, **hyperparams) 
    visualize_nlls_all(likelihood_frame)

def probe_uniform_abstract_template(language):
    def test_uat_selection(_likelihood_augmentation_frame, _subset_frame):
        for tag in _subset_frame['tag'].unique():
            subset_tag_frame = _subset_frame[_subset_frame['tag']==tag]
            subset_tag_likelihoods = sorted(subset_tag_frame['nll'], reverse=True)
            all_tag_frame = _likelihood_augmentation_frame[_likelihood_augmentation_frame['tag'] == tag]
            all_tag_likelihoods = sorted(all_tag_frame['nll'], reverse=True)
            for i in range(len(subset_tag_likelihoods)):
                assert subset_tag_likelihoods[i] == all_tag_likelihoods[i] 
        assert len(_likelihood_augmentation_frame) == 10000


    hyperparams = {'train_medium': True, 'use_empirical': False, 'num_aug': 2048}
    
    num_gold_test_examples = get_number_test_examples(language)
    likelihood_frame = load_augment_likelihoods(language, **hyperparams)
    initial_generation_frame = get_initial_generation_frame(language) # contains gold test + original 10,000 test examples.
    all_augmentation_frame = initial_generation_frame.loc[np.arange(num_gold_test_examples, len(initial_generation_frame))]

    augment_example_lengths = get_augmentation_example_lengths(initial_generation_frame, num_gold_test_examples)
    subset_sampler = get_subset_selecter(language, 'uat', get_initial_model_path(language, **hyperparams), initial_generation_frame, num_gold_test_examples, augment_example_lengths, **hyperparams)
    subset_augmentation_frame = subset_sampler.get_best_points(hyperparams['num_aug'])
    subset_augmentation_frame = subset_augmentation_frame.join(likelihood_frame)

    random_subset_augmentation_frame = all_augmentation_frame.sample(hyperparams['num_aug'], replace=False)
    random_subset_augmentation_frame['strategy'] = ['random'] * (hyperparams['num_aug'])
    subset_augmentation_frame['strategy'] = ['uat'] * (hyperparams['num_aug'])

    columns = ['tag', 'strategy']
    visualize_uat_selection(pd.concat([random_subset_augmentation_frame[columns], subset_augmentation_frame[columns]]), hyperparams['use_empirical'])

def main():
    hyperparams = {
        "num_aug": [1024],
        "train_medium": [True], 
        "use_high_loss": [True]
    }
    for language in LANGUAGES:
        for hparam_comb in generate_hyperparams(hyperparams):
            hparam_comb['r'] = 1
            hparam_comb['use_softmax_normalizer'] = False
            hparam_comb['use_high_loss'] = False 
            random_frame = inspect_augmentation_candidates(language, 'random', hparam_comb)
            uncertainty_frame = inspect_augmentation_candidates(language, 'uncertainty_sample', hparam_comb)
            print(random_frame['tag'].value_counts())
            print(uncertainty_frame['tag'].value_counts())

            # random_frame['strategy'] = ['random'] * len(random_frame)
            # uncertainty_frame['strategy'] = ['uncertainty_sample'] * len(uncertainty_frame)

            # all_augment_nll_frame = load_augment_likelihoods(language, **hparam_comb)
            # visualize_nll_comparison(pd.concat([random_frame, uncertainty_frame]).reset_index(), all_augment_nll_frame, f"{language}")
            # print(hparam_comb)
            # print(f"The average log uncertainty for random is {random_frame['nll'].apply(np.log).mean()} while it is {uncertainty_frame['nll'].apply(np.log).mean()} for uncertainty")

# visualize_uncertainty('bengali')    
# main()
probe_uniform_abstract_template('bengali')