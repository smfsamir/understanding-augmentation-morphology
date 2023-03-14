import pdb
import sys
import click
from typing import Dict
import seaborn as sns
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from Levenshtein import distance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from packages.fairseq_utils.dataloading_utils import get_initial_generation_frame, get_augmentation_example_lengths
from packages.utils.util_functions import get_number_test_examples, get_initial_model_path, generate_hyperparams
from packages.augmentation.subset_selecter_strategy import get_subset_selecter 
from packages.utils.constants import LANGUAGES, HALL_DATA_PATH, SCRATCH_PATH, SIGM_DATA_PATH, ANALYSIS_SCRATCH_PATH, ALGORITHMS, INITIAL_MODEL_PARAMS
from packages.visualizations.visualize import visualize_nll_comparison, visualize_nlls_all, visualize_uat_selection
from packages.pkl_operations.pkl_io import store_csv_dynamic

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
    initial_generation_frame = get_initial_generation_frame(language, hyperparams['aug_pool_size']) # contains gold test + original 10,000 test examples.
    num_gold_test_examples = get_number_test_examples(language)
    augment_example_lengths = get_augmentation_example_lengths(initial_generation_frame, num_gold_test_examples)
    subset_sampler = get_subset_selecter(language, augmentation_type, get_initial_model_path(language, **hyperparams), initial_generation_frame, num_gold_test_examples, augment_example_lengths, **hyperparams)
    # TODO: need to prefix with the number of points that are selected.
    subset_augmentation_frame = subset_sampler.get_best_points(hyperparams['num_aug'])
    subset_augmentation_frame = subset_augmentation_frame.reset_index()
    subset_augmentation_frame = subset_augmentation_frame.merge(likelihood_frame, left_on='index', right_index=True)
    return subset_augmentation_frame

def visualize_uncertainty(language):
    hyperparams = {"train_medium": True}
    likelihood_frame = load_augment_likelihoods(language, **hyperparams) 
    visualize_nlls_all(likelihood_frame)

def probe_uniform_abstract_template(language, aug_pool_size):
    def test_uat_selection(_likelihood_augmentation_frame, _subset_frame):
        for tag in _subset_frame['tag'].unique():
            subset_tag_frame = _subset_frame[_subset_frame['tag']==tag]
            subset_tag_likelihoods = sorted(subset_tag_frame['nll'], reverse=True)
            all_tag_frame = _likelihood_augmentation_frame[_likelihood_augmentation_frame['tag'] == tag]
            all_tag_likelihoods = sorted(all_tag_frame['nll'], reverse=True)
            for i in range(len(subset_tag_likelihoods)):
                assert subset_tag_likelihoods[i] == all_tag_likelihoods[i] 
        assert len(_likelihood_augmentation_frame) == aug_pool_size
        # TODO: this assertion will have to be changed if we vary the number of augmentations.


    hyperparams = {'train_medium': True, 'use_empirical': False, 'num_aug': 2048, 'aug_pool_size': aug_pool_size}
    
    num_gold_test_examples = get_number_test_examples(language)
    likelihood_frame = load_augment_likelihoods(language, **hyperparams)
    initial_generation_frame = get_initial_generation_frame(language, aug_pool_size) # contains gold test + original 10,000 test examples.
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

def inspect_nlls():
    hyperparams = {
        "num_aug": [128, 256, 512, 1024, 2048],
        "train_medium": [False], 
        "use_empirical": [False], 
        "use_loss": [True], 
        "rand_seed": [0],
        "aug_pool_size": [100000],
    }
    languages, random_nlls, uat_nlls, num_augs = [], [], [], []
    for language in LANGUAGES:
        print(f"Language: {language}")
        for hparam_comb in generate_hyperparams(hyperparams):
            random_frame = inspect_augmentation_candidates(language, 'random', hparam_comb)
            uat_frame = inspect_augmentation_candidates(language, 'uat', hparam_comb)
            # there is a column called 'nll'. Print the average of that for the random_frame and the uat_frame. Make sure to also print the number of examples ('num_aug')
            print(f"Random: {random_frame['nll'].mean()} ({len(random_frame)} examples)")
            print(f"UAT: {uat_frame['nll'].mean()} ({len(uat_frame)} examples)")
            languages.append(language)
            random_nlls.append(random_frame['nll'].mean())
            uat_nlls.append(uat_frame['nll'].mean())
            num_augs.append(len(random_frame))
    nll_frame = pd.DataFrame({
        "language": languages,
        "random": random_nlls,
        "uat_use_empirical=False_use_loss=True": uat_nlls,
        "num_aug": num_augs
    })
    nll_frame.to_csv(f"{SCRATCH_PATH}/nlls.csv", index=False)

@click.command()
def inspect_tag_distribution():
    hyperparams = {
        "num_aug": [128, 256, 512, 1024, 2048],
        "train_medium": [False], 
        "use_empirical": [False], 
        "use_loss": [True], 
        "rand_seed": [0],
        "aug_pool_size": [100000],
        "use_high_loss": [True]
    }
    strategies = ['random', 'uat', 'uncertainty_sample']
    tag_count_frames = []
    for language in LANGUAGES:
        print(f"Language: {language}")
        for hparam_comb in generate_hyperparams(hyperparams):
            for strategy in strategies:
                frame = inspect_augmentation_candidates(language, strategy, hparam_comb)
                # get the value counts of the tag column. Add a column called 'strategy' and set it to the strategy. Add a column called 'language' and set it to the language.
                value_counts = frame['tag'].value_counts()
                value_counts = value_counts.reset_index()
                value_counts.columns = ['tag', 'count']
                value_counts['strategy'] = strategy
                value_counts['language'] = language
                tag_count_frames.append(value_counts)

    tag_count_frame = pd.concat(tag_count_frames)
    tag_count_frame.to_csv(f"{SCRATCH_PATH}/tag_counts.csv", index=False)

def visualize_aug_tag_distribution():
    # iterate over all languages and load the augmentation frame. Then concatenate the the series of tags. Create another column called language to keep track of the language.
    # Then create a bar plot of the number of examples per tag for each language.

    # create a figure with len(LANGUAGES) rows and 1 column.
    # for each language, create a bar plot of the number of examples per tag.
    # save the figure to a file.
    fig, axes = plt.subplots(len(LANGUAGES), 1, figsize=(10, 10)) 
    for language in LANGUAGES:
        augmentation_frame = pd.read_csv(f"{HALL_DATA_PATH}/{language}-train-low-hall", header=None, names=["src", "tgt" ,"tag"], sep='\t')
        # use seaborn to create a bar plot of the number of examples per tag.
        # use a label encoder to convert the tag to a number.
        le = LabelEncoder()
        augmentation_frame['tag'] = le.fit_transform(augmentation_frame['tag'])
        tag_series = augmentation_frame['tag']
        tag_series = tag_series.value_counts()
        ax = axes[LANGUAGES.index(language)]
        sns.barplot(x=tag_series.index, y=tag_series.values, ax=ax)
        ax.set_title(language)
        # print the number of tags in the series.
        print(f"{language}: {len(tag_series)}")
    plt.savefig("tag_distribution.png")

def compute_proportion_perturbed(language):
    augmentation_frame = pd.read_csv(f"{HALL_DATA_PATH}/{language}-train-low-hall-100000", header=None, names=["src", "tgt" ,"tag", "candidate_inds", "source_key"], sep='\t')
    gt_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-train-low", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    # convert the candidate_inds column to a list of ints. Currently, it is a list of ints that has been converted to a string.
    augmentation_frame['candidate_inds'] = augmentation_frame['candidate_inds'].apply(lambda x: [int(i) for i in x[1:-1].split(',')])


    # merge the two frames using the source_key column for the augmentation frame and the index for the gt_frame.
    # then for each of the 100,000 examples, iterate over the candidate_inds column using a variable i. If the tgt[i] != gt_frame['tgt'][i], then increment a counter. Afterwards, normalize the counter by the number of candidate_inds. Add a column to the augmentation_frame called 'proportion_perturbed' and set it to the normalized counter.
    merged_frame = augmentation_frame.merge(gt_frame, left_on='source_key', right_index=True)
    def compute_proportion_perturbed_helper(row):
        counter = 0
        for i in row['candidate_inds']:
            try:
                if row['tgt_x'][i] != row['tgt_y'][i]:
                    counter += 1
            except:
                print(row['tgt_x'])
                print(row['tgt_y'])
                print(i)
                pdb.set_trace()
    merged_frame['proportion_perturbed'] = merged_frame.apply(lambda row: compute_proportion_perturbed_helper(row), axis=1)
    # print the mean of the proportion_perturbed column.
    return (merged_frame['proportion_perturbed'].mean())

# for language in LANGUAGES:
#     print(f"Proportion perturbed for {language}: {compute_proportion_perturbed(language)}")

@click.command()
@click.argument("aug_pool_size", type=int)
def inspect_high_loss_candidates(aug_pool_size):
    # open the likelihoods path and load it into a dataframe.
    hyperparams = {
        "rand_seed": 0,
        "aug_pool_size": 100000, 
        "train_medium": False
    }
    for language in LANGUAGES:
        initial_path = get_initial_model_path(language, **hyperparams)
        likelihood_pkl_path = f"{initial_path}/{language}_log_likelihoods.pickle"
        with open(likelihood_pkl_path, 'rb') as f:
            example_likelihoods = pkl.load(f)
        indices, log_likelihoods = zip(*example_likelihoods)
        negative_log_likelihoods = -np.array(log_likelihoods)
        likelihood_frame = pd.DataFrame({
            "nll": negative_log_likelihoods
        }, index = indices)
        generation_frame = get_initial_generation_frame(language, aug_pool_size)
        num_generation = len(generation_frame)
        num_gold = num_generation - aug_pool_size
        # load the train frame for the language.
        train_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-train-low", header=None, names=["src", "tgt" ,"tag"], sep='\t')
        augmentation_frame = generation_frame.loc[np.arange(num_gold, num_generation)] 
        augmentation_frame = augmentation_frame.join(likelihood_frame)

        # use the test_frame and the test_foreign_key column in augmentation_frame to add a column original_tgt to augmentation_frame.
        train_frame = train_frame.rename(columns={"tgt": "original_tgt"})
        augmentation_frame = augmentation_frame.merge(train_frame[['original_tgt']], left_on='test_foreign_key', right_index=True)
        print(augmentation_frame)

        # # compute the levenstein distance between the original_tgt and the tgt column in augmentation_frame. Add a column called levenstein_distance to augmentation_frame.
        augmentation_frame['levenstein_distance'] = augmentation_frame.apply(lambda row: distance(row['original_tgt'], row['tgt']), axis=1)
        # add another column proportion_perturbed to augmentation_frame. 
        print(augmentation_frame)
        # store the augmentation frame to a csv file in the ANALYSIS_SCRATCH_PATH.
        store_csv_dynamic(augmentation_frame, "loss_analysis_frame_{}_{}".format(language, aug_pool_size), ANALYSIS_SCRATCH_PATH)

@click.group()
def main():
    pass

main.add_command(inspect_high_loss_candidates)
main.add_command(inspect_tag_distribution)

if __name__ == "__main__":
    main()