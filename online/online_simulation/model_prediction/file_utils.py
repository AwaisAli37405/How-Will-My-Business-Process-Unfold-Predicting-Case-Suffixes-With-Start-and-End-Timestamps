import random
import os
import json
import copy
from datetime import timedelta
import pandas as pd
import numpy as np
import configparser as cp
import itertools
import readers.log_reader as lr
import utils.support as sup
import jellyfish as jf
from tensorflow.keras.models import load_model





def split_log(log):
    key = 'start_timestamp'

    # Sort the log by the relevant timestamp
    log = log.sort_values(by=key)

    # Determine the split point (80% for suffix, 20% for prefix)
    split_index = int(len(log) * 0.3)

    # Split the data
    prefixes = log.iloc[:split_index]
    suffixes = log.iloc[split_index:]

    cutt_off = prefixes[key].max()
    print("cutt_off:", cutt_off)

    return prefixes, suffixes, cutt_off


def _scale_inter(log, add_cols, norm_method, args):
    # log, scale_args = self.scale_feature(log, 'dur', self.norm_method)

    log = scale_feature(log, 'dur', method=norm_method, scale_args=args['dur'])
    log = scale_feature(log, 'wait', method=norm_method, scale_args=args['wait'])

    for col in add_cols:
        if col == 'daytime':
            log = scale_feature(log, 'daytime', 'day_secs', args['daytime'])
        elif col == 'weekday':
            continue
        else:
            log = scale_feature(log, col, norm_method, args[col])
    return log

# =========================================================================
# Scale features
# =========================================================================

def scale_feature(log, feature, method,scale_args, replace=False):


    if method == 'lognorm':
        log[feature + '_log'] = np.log1p(log[feature])
        max_value = scale_args['max_value']
        min_value = scale_args['min_value']
        log[feature+'_norm'] = np.divide(
                np.subtract(log[feature+'_log'], min_value), (max_value - min_value))
        log = log.drop((feature + '_log'), axis=1)

    elif method == 'normal':
        max_value = scale_args['max_value']
        min_value = scale_args['min_value']
        log[feature+'_norm'] = np.divide(
                np.subtract(log[feature], min_value), (max_value - min_value))

    elif method == 'standard':
        mean = scale_args['mean']
        std = scale_args['std']
        log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
                                           std)

    elif method == 'max':
        max_value = scale_args['max_value']
        log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                  if max_value > 0 else 0)

    elif method == 'day_secs':
        max_value = 86400
        log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                  if max_value > 0 else 0)

    elif method is None:
        log[feature+'_norm'] = log[feature]
    else:
        raise ValueError(method)
    if replace:
        log = log.drop(feature, axis=1)
    return log

# =========================================================================

def scale(value, parms, scale_args):
    if parms['norm_method'] == 'lognorm':
        # Apply logarithmic normalization
        value = np.log1p(value)
        max_value = scale_args['max_value']
        min_value = scale_args['min_value']
        # Check if denominator is not zero
        if (max_value - min_value) != 0:
            value = (value - min_value) / (max_value - min_value)
        else:
            value = 0  # or handle as appropriate
    elif parms['norm_method'] == 'normal':
        # Scale the value to a range between 0 and 1
        max_value = scale_args['max_value']
        min_value = scale_args['min_value']
        # Check if denominator is not zero
        if (max_value - min_value) != 0:
            value = (value - min_value) / (max_value - min_value)
        else:
            value = 0  # or handle as appropriate
    elif parms['norm_method'] == 'standard':
        # Apply standardization (z-score normalization)
        mean = scale_args['mean']
        std = scale_args['std']
        # Check if std is not zero
        if std != 0:
            value = (value - mean) / std
        else:
            value = 0  # or handle as appropriate
    elif parms['norm_method'] == 'max':
        # Scale the value by dividing by the maximum value
        max_value = scale_args['max_value']
        # Check if max_value is not zero
        if max_value != 0:
            value = value / max_value
        else:
            value = 0  # or handle as appropriate
    elif parms['norm_method'] is None:
        # Return the value as is
        value = value
    else:
        raise ValueError(parms['norm_method'])
    return value



def rescale(value, parms, scale_args):
    if parms['norm_method'] == 'lognorm':
        max_value = scale_args['max_value']
        min_value = scale_args['min_value']
        value = (value * (max_value - min_value)) + min_value
        # Maximum value chosen to prevent overflow in expm1
        max_safe_value = 709  # expm1(709) ~ largest float64 representation
        value = np.clip(value, -max_safe_value, max_safe_value)
        value = np.expm1(value)
    elif parms['norm_method'] == 'normal':
        max_value = scale_args['max_value']
        min_value = scale_args['min_value']
        value = (value * (max_value - min_value)) + min_value
    elif parms['norm_method'] == 'standard':
        mean = scale_args['mean']
        std = scale_args['std']
        value = (value * std) + mean
    elif parms['norm_method'] == 'max':
        max_value = scale_args['max_value']
        value = np.rint(value * max_value)
    elif parms['norm_method'] is None:
        value = value
    else:
        raise ValueError(parms['norm_method'])
    return value


def sort_event_log(event_log_, case_column, start_timestamp_column, activity_column):
    """
    Sorts an event log by the start timestamp, ensuring the 'start' event is the first in each case.

    Args:
        event_log (pd.DataFrame): The event log as a DataFrame.
        case_column (str): The column representing the Case ID.
        start_timestamp_column (str): The column representing the start timestamp.
        activity_column (str): The column representing activities.

    Returns:
        pd.DataFrame: The sorted event log.
    """
    # Ensure the start_timestamp is in datetime format

    event_log = event_log_.copy()
    event_log.loc[:, start_timestamp_column] = pd.to_datetime(event_log[start_timestamp_column])


    # Define a helper function to sort each case
    def sort_case(case_group):
        # Mark the start event as first
        # case_group['sort_order'] = case_group[activity_column].apply(lambda x: 0 if x == "start" else 1)
        # Sort within the case by sort_order and start_timestamp
        case_group = case_group.sort_values(by=[activity_column, start_timestamp_column])
        # Drop the auxiliary column
        # case_group.drop(columns=['sort_order'], inplace=True)
        return case_group

    # Apply sorting to each case
    sorted_log = (
        event_log.groupby(case_column, group_keys=False)
        .apply(sort_case)
        .sort_values(by=[case_column, start_timestamp_column])  # Sort overall by CaseID and timestamp
    )

    return sorted_log

def insert_case_start_and_end_events(log, start_act, end_act, rl_start, rl_end):

    log = log.sort_values(by=['caseid', 'start_timestamp'])

    # Create an empty list to store rows with added start and end events
    modified_log = []

    case_column = 'caseid'
    start_timestamp_column = 'start_timestamp'
    end_timestamp_column = 'end_timestamp'
    ac_index_column = 'ac_index'
    rl_index_column = 'rl_index'


    # List of all columns in the event log except case, activity, start timestamp, and end timestamp
    other_columns = [col for col in log.columns if
                     col not in {case_column, start_timestamp_column, end_timestamp_column,
                                 ac_index_column, rl_index_column}]

    # Group by case ID
    for case_id, group in log.groupby('caseid'):
        group = group.sort_values(by='start_timestamp')  # Ensure events are sorted

        # Extract the earliest start and latest end timestamps
        case_start_time = group['start_timestamp'].min()
        case_end_time = group['end_timestamp'].max()

        # Initialize default values for the other columns
        zero_values = {col: 0 for col in other_columns}

        # Add case-level start event
        modified_log.append({
            case_column: case_id,
            start_timestamp_column: case_start_time,
            end_timestamp_column: case_start_time,
            ac_index_column : start_act,
            rl_index_column : rl_start,
            **zero_values
        })

        # Add the original events
        modified_log.extend(group.to_dict('records'))

        # Add case-level end event
        modified_log.append({
            case_column: case_id,
            start_timestamp_column: case_end_time,
            end_timestamp_column: case_end_time,
            ac_index_column: end_act,
            rl_index_column: rl_end,
            **zero_values
        })

    # Convert the modified log back to a DataFrame
    modified_log_df = pd.DataFrame(modified_log)

    # Ensure the log is sorted correctly
    modified_log_df = modified_log_df.sort_values(by=[case_column, start_timestamp_column])

    return modified_log_df



def reformat_events( log, columns,ac_index, rl_index, prefix):
    """Creates series of activities, roles and relative times per trace.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    temp_data = list()
    log_df = log.to_dict('records')
    key = 'start_timestamp'
    log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
    for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        temp_dict = dict()
        for x in columns:
            serie = [y[x] for y in trace]
            if x == 'ac_index':
                if prefix:
                    serie.insert(0, ac_index[('start')])
                if prefix == None:
                    serie.insert(0, ac_index[('start')])
                    serie.append(ac_index[('end')])
                # else:
                #     serie.append(self.ac_index[('end')])
            elif x == 'rl_index':
                if prefix:
                    serie.insert(0, rl_index[('start')])
                if prefix == None:
                    serie.insert(0, rl_index[('start')])
                    serie.append(rl_index[('end')])
                # else:
                #     serie.append(self.rl_index[('end')])

            # elif x == 'caseid':
            #     serie.insert(0, ])

            else:
                if prefix:
                    serie.insert(0, 0)
                if prefix == None:
                    serie.insert(0, 0)
                    serie.append(0)
                # else:
                #     serie.append(0)
            temp_dict = {**{x: serie}, **temp_dict}
        temp_dict = {**{'caseid': key}, **temp_dict}
        temp_data.append(temp_dict)
    return temp_data


def load_parameters(output_route,selfparms):
    # Loading of parameters from training
    path = os.path.join(output_route,
                        'parameters',
                        'model_parameters.json')
    with open(path) as file:
        data = json.load(file)
        if 'activity' in data:
            del data['activity']
        parms = {k: v for k, v in data.items()}
        parms.pop('rep', None)
        distinct_parms = {**selfparms, **parms}
        if 'dim' in data.keys():
            distinct_parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
        if selfparms['one_timestamp']:
            distinct_parms['scale_args'] = {
                k: float(v) for k, v in data['scale_args'].items()}
        else:
            for key in data['scale_args'].keys():
                distinct_parms['scale_args'][key] = {
                    k: float(v) for k, v in data['scale_args'][key].items()}
        distinct_parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
        distinct_parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
        file.close()
        ac_index = {v: k for k, v in distinct_parms['index_ac'].items()}
        rl_index = {v: k for k, v in distinct_parms['index_rl'].items()}

        return distinct_parms, ac_index, rl_index
