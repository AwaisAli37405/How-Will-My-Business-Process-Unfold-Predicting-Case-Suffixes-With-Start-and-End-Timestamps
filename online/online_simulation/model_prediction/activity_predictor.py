"""
Author: awaisali
Date: 16.11.2024
Email: m.awaisali.mirza@gmail.com
Project: Deep_Learning_Simulator
"""

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

from model_training.features_manager import FeaturesMannager as feat
from model_prediction.file_utils import _scale_inter
from model_prediction.file_utils import rescale
from model_prediction.file_utils import reformat_events
from model_prediction.file_utils import load_parameters
from model_prediction.file_utils import sort_event_log

class ActivityPredictor():

    def __init__(self, parms):
        self.output_route_act_pred = os.path.join('output_files', parms['activity_predictor'])

        self.parms = parms
        # load parameters
        self.parms, self.ac_index, self.rl_index = load_parameters(
            self.output_route_act_pred, self.parms)

        self.one_timestamp = self.parms['one_timestamp']

    def get_ac_rl_index(self):
        return self.ac_index, self.rl_index, self.parms['max_trace_size']

    def load_model(self):
        model_act = load_model(os.path.join(self.output_route_act_pred,
                                            self.parms['model_file']),compile=False)

        return model_act



    def predict(self, log, model):
        # log = feat.add_calculated_times(self, log)

        # prepare the log for prediction
        # log = _scale_inter(log, [], self.parms['norm_method'], self.parms['scale_args'])

        # now initializing the ols used in predicting the next duration

        cols = ['caseid',  'end_timestamp', 'start_timestamp',
                'ac_index', 'rl_index', 'active_cases_len', 'available_resources',
                'dur_norm', 'wait_norm', 'dur','wait']

        reformatted_events = reformat_events(log, cols,self.ac_index, self.rl_index, prefix=True)

        # now creating the samples
        for event in reformatted_events:
            case = event['caseid'][1]
            trace = log[log['caseid'] == case]

            trace = sort_event_log(trace, case_column='caseid', start_timestamp_column='start_timestamp',
                                   activity_column='ac_index')

            trace = trace.drop_duplicates(subset=['caseid',  'ac_index', 'rl_index',
                                                  'start_timestamp'])  # because there can be multiple batches of the same case# because there can be multiple batches of the same case
            # trace = trace.iloc[0].to_frame().T
            vec = self.create_samples(trace, col=['wait_norm', 'active_cases_len', 'available_resources',
                                                  'ac_index', 'rl_index', 'dur_norm'])  # trace.iloc[0].to_frame().T

            # now predicting the duration
            print(f"Predicting  case {case}")
            pred = self.predict_case(vec, model)


            record = self.create_new_activity_instance(pred, log.columns.tolist(), case, trace)
            log = log.append(record, ignore_index=True)



        return log

    def predict_case(self, prefix, model):

        self.parms['dim']['time_dim'] = model.input_shape[0][1]
        # Create an array of zeros with the same length as the time dimension
        zeros_array = np.zeros(self.parms['dim']['time_dim'])

        # Convert the 'activities' prefix to a numpy array
        activities_array = np.array(prefix['activities'])

        # Append the activities array to the zeros array along the first axis
        appended_array = np.append(zeros_array, activities_array, axis=0)

        # Take the last 'time_dim' elements from the appended array
        trimmed_array = appended_array[-self.parms['dim']['time_dim']:]

        # Reshape the trimmed array to have shape (1, time_dim)
        x_ac_ngram = trimmed_array.reshape((1, self.parms['dim']['time_dim']))

        x_rl_ngram = (np.append(
            np.zeros(self.parms['dim']['time_dim']),
            np.array(prefix['roles']),
            axis=0)[-self.parms['dim']['time_dim']:]
                      .reshape((1, self.parms['dim']['time_dim'])))

        # times input shape(1,5,1)
        times_attr_num = (prefix['times'][0].shape[0])
        x_t_ngram = np.array(
            [np.append(np.zeros(
                (self.parms['dim']['time_dim'], times_attr_num)),
                prefix['times'], axis=0)
             [-self.parms['dim']['time_dim']:]
             .reshape((self.parms['dim']['time_dim'], times_attr_num))]
        )

        # inter_attr_num = (prefix['inter_attr'][0].shape[0])
        # x_inter_ngram = np.array(
        #     [np.append(np.zeros((
        #         self.parms['dim']['time_dim'], inter_attr_num)),
        #         prefix['inter_attr'], axis=0)
        #     [-self.parms['dim']['time_dim']:]
        #     .reshape(
        #         (self.parms['dim']['time_dim'], inter_attr_num))]
        # )

        # next_event = (np.append(
        #     np.zeros(self.parms['dim']['time_dim']),
        #     np.array(prefix['next_evt_act']),
        #     axis=0)[-self.parms['dim']['time_dim']:]
        #               .reshape((1, self.parms['dim']['time_dim'])))
        #
        # next_event_start = (np.append(
        #     np.zeros(self.parms['dim']['time_dim']),
        #     np.array(prefix['next_evt_start']),
        #     axis=0)[-self.parms['dim']['time_dim']:]
        #                     .reshape((1, self.parms['dim']['time_dim'])))

        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]

        preds = model.predict(inputs)


        return preds

    def create_samples(self, log_, col):

        times = ['wait_norm', 'dur_norm']
        equi = {'ac_index': 'activities', 'rl_index': 'roles'}
        # vec = {'prefixes': dict()}
        vec = {}
        x_weekday = list()
        y_weekday = list()
        # times
        x_times_dict = dict()
        y_times_dict = dict()
        # intercases
        x_inter_dict = dict()

        # log = traces.iloc[-1]

        if len(log_) > 1:
            log = log_.copy()

            for x in col:
                # serie = [log[i][x]]
                serie = log[x].tolist()

                if x in list(equi.keys()):
                    vec[equi[x]] = serie
                elif x in times:
                    x_times_dict[x] = serie
                elif x == 'weekday':
                    x_weekday = serie

                else:
                    x_inter_dict[x] = serie

            vec['times'] = list()
            x_times_dict = pd.DataFrame(x_times_dict)
            for row in x_times_dict.values:
                new_row = np.array(row)
                # new_row = np.dstack(new_row)
                # new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
                vec['times'].append(new_row)
            # Reshape intercase expected attributes (prefixes, # attributes)

            vec['inter_attr'] = list()
            x_inter_dict = pd.DataFrame(x_inter_dict)
            for row in x_inter_dict.values:
                new_row = np.array(row)
                # new_row = np.dstack(new_row)
                # new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
                vec['inter_attr'].append(new_row)

            # vec['next_evt_act'] = list()
            # vec['next_evt_act'].append(log_['ac_index'].values[-1])
            # vec['next_evt_start'] = list()
            # vec['next_evt_start'].append(log_['wait_norm'].values[-1])

        else:
            log = log_.copy()
            for x in col:
                # serie = [log[i][x]]
                serie = log[x].tolist()

                if x in list(equi.keys()):
                    if x in 'ac_index':
                        vec[equi[x]] = [self.ac_index[('start')]]
                    elif x in 'rl_index':
                        vec[equi[x]] = [self.rl_index[('start')]]
                elif x in 'wait_norm':
                    x_times_dict[x] = [0]

                elif x in 'dur_norm':
                    x_times_dict[x] = [0]
                elif x == 'weekday':
                    x_weekday = serie

                else:
                    x_inter_dict[x] = serie

            vec['times'] = list()
            x_times_dict = pd.DataFrame(x_times_dict)
            for row in x_times_dict.values:
                new_row = np.array(row)
                # new_row = np.dstack(new_row)
                # new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
                vec['times'].append(new_row)
            # Reshape intercase expected attributes (prefixes, # attributes)

            vec['inter_attr'] = list()
            x_inter_dict = pd.DataFrame(x_inter_dict)
            for row in x_inter_dict.values:
                new_row = np.array(row)
                # new_row = np.dstack(new_row)
                # new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
                vec['inter_attr'].append(new_row)

            # vec['next_evt_act'] = list()
            # vec['next_evt_act'].append(log['ac_index'].values[0])
            # vec['next_evt_start'] = list()
            # vec['next_evt_start'].append(log_['wait_norm'].values[0])

        return vec

    def create_new_activity_instance(self, pred, cols,case, trace):

        modified_log = []

        case_column = 'caseid'

        start_timestamp_column = 'start_timestamp'
        end_timestamp_column = 'end_timestamp'
        ac_index_column = 'ac_index'
        rl_index_column = 'rl_index'

        other_columns = [col for col in cols if
                         col not in {case_column, start_timestamp_column, end_timestamp_column,
                                     ac_index_column, rl_index_column}]

        # Initialize default values for the other columns
        zero_values = {col: 0 for col in other_columns}

        # role = next(key for key, value in self.rl_index.items() if value == np.argmax(pred[1][0]))
        # act = next(key for key, value in self.ac_index.items() if value == np.argmax(pred[0][0]))

        # Determine activity and role indices based on the variant
        if self.parms['variant'] == 'rc':
            # Random choice based on probability distribution
            pos = np.random.choice(np.arange(len(pred[0][0])), p=pred[0][0])
            pos1 = np.random.choice(np.arange(len(pred[1][0])), p=pred[1][0])
        elif self.parms['variant'] == 'am':
            # Argmax for deterministic selection
            pos = np.argmax(pred[0][0])
            pos1 = np.argmax(pred[1][0])
        elif self.parms['variant'] == 'da':
            # Frequency-adjusted probability calculation for activity index
            counts = {int(num): trace[ac_index_column].tolist().count(int(num)) for num in trace[ac_index_column]}
            if counts:
                total_weighted_prob = sum(pred[0][0][key] * count for key, count in counts.items())
                for x in range(len(pred[0][0])):
                    if x in counts and counts[x] > 0:
                        pred[0][0][x] /= counts[x] * total_weighted_prob
                pos = np.argmax(pred[0][0])
            else:
                pos = np.argmax(pred[0][0])

            # Frequency-adjusted probability calculation for role index
            if not trace[rl_index_column].empty:
                time_counts = {int(num): trace[rl_index_column].tolist().count(int(num)) for num in
                               trace[rl_index_column]}
                total_weighted_prob_time = sum(pred[1][0][key] * count for key, count in time_counts.items())
                for x in range(len(pred[1][0])):
                    if x in time_counts and time_counts[x] > 0:
                        pred[1][0][x] /= time_counts[x] * total_weighted_prob_time
                pos1 = np.argmax(pred[1][0])
            else:
                pos1 = np.argmax(pred[1][0])
        else:
            raise ValueError(f"Unsupported variant: {self.parms['variant']}")




        # Add case-level start event
        modified_log.append({
            case_column: case,
            start_timestamp_column: None,
            end_timestamp_column: None,
            ac_index_column: pos,
            rl_index_column: pos1,
            **zero_values
        })

        modified_log_df = pd.DataFrame(modified_log)

        return modified_log_df

