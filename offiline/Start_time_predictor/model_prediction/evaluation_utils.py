"""
Author: awaisali
Date: 07.10.2024
Email: m.awaisali.mirza@gmail.com
Project: Deep_Learning_Simulator
"""
import jellyfish as jf
import itertools
from statistics import mean
import random
import numpy as np
import pandas as pd


def evalaute_simulation(predictions):
    data = predictions.copy()
    diff_dur = []
    diff_wait = []
    for list in data:
        n = min(len(list['dur_expect']), len(list['dur_pred']), len(list['wait_expect']), len(list['wait_pred']))
        for i in range(n):
            diff_dur.append(abs(list['dur_expect'][i] - list['dur_pred'][i]))
            diff_wait.append(abs(list['wait_expect'][i] - list['wait_pred'][i]))

    dur_mae = mean(diff_dur)
    wait_mae = mean(diff_wait)

    return dur_mae, wait_mae



def create_task_alias(categories):
    """
    Create string alias for tasks names or tuples of tasks-roles names

    Parameters
    ----------
    features : list

    Returns
    -------
    alias : alias dictionary

    """
    variables = sorted(categories)
    characters = [chr(i) for i in range(0, len(variables))]
    aliases = random.sample(characters, len(variables))
    alias = dict()
    for i, _ in enumerate(variables):
        alias[variables[i]] = aliases[i]
    return alias


def _similarity_evaluation(data, feature):
    data = data.copy()

    data = data[[(feature + '_expect'), (feature + '_pred')]]

    # append all values and create alias
    values = (data[feature + '_pred'].tolist() +
              data[feature + '_expect'].tolist())
    values = list(set(itertools.chain.from_iterable(values)))
    index = create_task_alias(values)
    for col in ['_expect', '_pred']:
        list_to_string = lambda x: ''.join([index[y] for y in x])
        data['suff' + col] = (data[feature + col]
                              .swifter.progress_bar(False)
                              .apply(list_to_string))

    # measure similarity between pairs

    def distance(x, y):
        return (1 - (jf.damerau_levenshtein_distance(x, y) /
                     np.max([len(x), len(y)])))

    data['similarity'] = (data[['suff_expect', 'suff_pred']]
                          .swifter.progress_bar(False)
                          .apply(lambda x: distance(x.suff_expect,
                                                    x.suff_pred), axis=1))

    mean = data['similarity'].mean()

    return mean


def _mae_remaining_evaluation(data, feature):
    data = data.copy()

    # data = data[[(feature + '_expect'), (feature + '_pred'),
    #              'run_num', 'implementation', 'pref_size']]
    data = data[[(feature + '_expect'), (feature + '_pred')]]
    ae = (lambda x: np.abs(np.sum(x[feature + '_expect']) -
                           np.sum(x[feature + '_pred'])))
    data['ae'] = data.apply(ae, axis=1)
    mean = data['ae'].mean()

    return mean
