"""
Author: awaisali
Date: 16.11.2024
Email: m.awaisali.mirza@gmail.com
Project: Deep_Learning_Simulator
"""
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
from model_prediction.file_utils import sort_event_log
from model_prediction.file_utils import reformat_events
from model_prediction.file_utils import load_parameters
from model_prediction.file_utils import scale



class DurPredictor():

    def __init__(self, parms):
        self.output_route_act_pred = os.path.join('output_files', parms['dur_predictor'])

        self.parms = parms
        # load parameters
        self.parms, self.ac_index, self.rl_index = load_parameters(
            self.output_route_act_pred, self.parms)

        self.one_timestamp = self.parms['one_timestamp']


    def trace_completion(self, log, resource_count):

        log = log.sort_values(by='start_timestamp')

        # now calculating the inter case features
        log = feat.get_resource_availability(log, self.parms, resource_count)
        log = feat.get_inter_arrival_time_case(log)

        return log

    def predict(self, log, model):
        log = log.reset_index(drop=True)
        print("length of active cases:", len(log['caseid'].unique()))

        # Define the threshold for 1 month in seconds

        log = feat.add_calculated_times(self, log)

        # prepare the log for prediction
        log = _scale_inter(log, [], self.parms['norm_method'], self.parms['scale_args'])

        # now initializing the ols used in predicting the next duration

        cols = ['caseid',  'end_timestamp', 'start_timestamp',
                'ac_index', 'rl_index', 'active_cases_len', 'available_resources','inter_arrival_time',
                'dur_norm', 'wait_norm', 'dur','wait']

        reformatted_events = reformat_events(log, cols, self.ac_index, self.rl_index, prefix=True)

        for event in reformatted_events:
            # Extract the case ID
            case = event['caseid'][1]
            print(f"Predicting for case {case}")

            # Filter the trace for the specific case ID
            trace = log[log['caseid'] == case]

            # Sort the event log for the case
            trace = sort_event_log(trace, case_column='caseid', start_timestamp_column='start_timestamp',
                                   activity_column='ac_index')

            # Remove duplicate events for the same case
            trace = trace.drop_duplicates(subset=['caseid', 'ac_index', 'rl_index', 'start_timestamp'])

            # Loop to handle and fill all NaT occurrences
            while trace['end_timestamp'].isna().sum() > 0:
                # Count rows where 'end_timestamp' is NaT
                nat_count = trace['end_timestamp'].isna().sum()
                print("NaT count:", nat_count)

                if nat_count > 0:

                    trace.sort_index(inplace=True)
                    first_nat_index = trace['end_timestamp'].isna().idxmax()

                    trace_trimmed = trace[trace.index < (first_nat_index+1)]
                    # Generate input samples for prediction
                    vec = self.create_samples(trace_trimmed,
                                              col=['wait_norm', 'active_cases_len', 'available_resources','inter_arrival_time',
                                                   'ac_index', 'rl_index', 'dur_norm'])

                    # Predict the duration
                    print(f"Predicting duration for case {case}")
                    pred = self.predict_case(vec, model)
                    # Check if the prediction is NaN
                    # if np.isnan(pred[0][0]):  # Use np.isnan to check for NaN
                    #     print(f"Predicted value is NaN for case {case}. Setting to 0.")
                    #     pred[0][0] = 0  # Set NaN predictions to 0
                    #
                    # print(f"Predicted duration: {pred[0][0]} for case {case}")



                    # Get the index of the row to update
                    trace_index = trace_trimmed.index[-1]


                    # Update duration in normalized and original scales

                    rescaled_dur = rescale(pred[0][0], self.parms, self.parms['scale_args']['dur'])
                    # print(f"Rescaled duration: {rescaled_dur}")



                    # Ensure rescaled_dur is not NaN and within a valid range
                    if not pd.isna(rescaled_dur):
                        if 0 <= rescaled_dur <= np.max(log['dur']):
                            log.loc[trace_index, 'end_timestamp'] = log.loc[trace_index, 'start_timestamp'] + timedelta(seconds=rescaled_dur)
                            log.loc[trace_index, 'dur_norm'] = pred[0][0]
                            log.loc[trace_index, 'dur'] = rescaled_dur
                        elif rescaled_dur > np.max(log['dur']):
                            rescaled_dur = log['dur'].mean(skipna=True)  # Preferred in Pandas
                            log.loc[trace_index, 'end_timestamp'] = log.loc[trace_index, 'start_timestamp'] + timedelta(
                                seconds=rescaled_dur)
                            rescaled = scale(rescaled_dur, self.parms, self.parms['scale_args']['dur'])
                            log.loc[trace_index, 'dur_norm'] = rescaled
                            log.loc[trace_index, 'dur'] = rescaled_dur
                        elif rescaled_dur < 0:
                            rescaled_dur = 0
                            log.loc[trace_index, 'end_timestamp'] = log.loc[trace_index, 'start_timestamp'] + timedelta(
                                seconds=rescaled_dur)
                            rescaled = scale(rescaled_dur, self.parms, self.parms['scale_args']['dur'])
                            log.loc[trace_index, 'dur_norm'] = rescaled
                            log.loc[trace_index, 'dur'] = rescaled_dur


                    # print(f"Trace after update: {log.loc[trace_index][['caseid','dur_norm', 'dur', 'end_timestamp']]}")

                    # Refresh the trace to include updated values
                    trace = log[log['caseid'] == case]

                    # Sort and deduplicate again after updates (optional but ensures consistency)
                    trace = sort_event_log(trace, case_column='caseid', start_timestamp_column='start_timestamp',
                                           activity_column='ac_index')
                    trace = trace.drop_duplicates(subset=['caseid', 'ac_index', 'rl_index', 'start_timestamp'])


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

        inter_attr_num = (prefix['inter_attr'][0].shape[0])
        x_inter_ngram = np.array(
            [np.append(np.zeros((
                self.parms['dim']['time_dim'], inter_attr_num)),
                prefix['inter_attr'], axis=0)
            [-self.parms['dim']['time_dim']:]
            .reshape(
                (self.parms['dim']['time_dim'], inter_attr_num))]
        )

        next_event = (np.append(
            np.zeros(self.parms['dim']['time_dim']),
            np.array(prefix['next_evt_act']),
            axis=0)[-self.parms['dim']['time_dim']:]
                      .reshape((1, self.parms['dim']['time_dim'])))

        next_event_start = (np.append(
            np.zeros(self.parms['dim']['time_dim']),
            np.array(prefix['next_evt_start']),
            axis=0)[-self.parms['dim']['time_dim']:]
                            .reshape((1, self.parms['dim']['time_dim'])))

        inputs = [x_ac_ngram, x_rl_ngram, next_event, next_event_start, x_t_ngram, x_inter_ngram]

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
            log = log_.iloc[:-1]

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

            vec['next_evt_act'] = list()
            vec['next_evt_act'].append(log_['ac_index'].values[-1])
            vec['next_evt_start'] = list()
            vec['next_evt_start'].append(log_['wait_norm'].values[-1])

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

            vec['next_evt_act'] = list()
            vec['next_evt_act'].append(log['ac_index'].values[0])
            vec['next_evt_start'] = list()
            vec['next_evt_start'].append(log_['wait_norm'].values[0])

        return vec



    def load_model(self):
        model_act = load_model(os.path.join(self.output_route_act_pred,
                                            self.parms['model_file']),compile=False)

        return model_act


