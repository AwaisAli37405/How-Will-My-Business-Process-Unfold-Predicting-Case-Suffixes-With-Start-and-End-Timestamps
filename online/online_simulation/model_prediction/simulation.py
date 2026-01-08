"""
Author: awaisali
Date: 29.09.2024
Email: m.awaisali.mirza@gmail.com
Project: Deep_Learning_Simulator
"""


import pandas as pd
import itertools
import numpy as np
import copy
import gc
from model_prediction import file_utils
from model_training.features_manager import FeaturesMannager as feat
from model_prediction.file_utils import insert_case_start_and_end_events


class Simulation:

    def __init__(self, obj):
        # self.log = obj.log
        self.obj_act_pred = obj.act_pred
        self.obj_dur_pred = obj.dur_pred
        self.obj_start_pred = obj.start_pred

        self.log = obj.log_test
        # self.log = self.log[self.log['caseid'].isin(self.log['caseid'].unique()[:2])]
        self.cut_off = obj.cutt_off
        self.get_start_end()

        self.LoadModel()

        self.g_truth = {}
        self.predictions = {}
        self.resource_count = obj.resource_counts
        # self.cols = define_columns(None)

        # self.simulate() # to start the simulation

    def get_start_end(self):

        ac_index, rl_index, max_len = self.obj_act_pred.get_ac_rl_index()
        self.start_act = ac_index['start']
        self.end_act = ac_index['end']
        self.rl_start = rl_index['start']
        self.rl_end = rl_index['end']
        self.max_len = max_len
        self.log = insert_case_start_and_end_events(self.log, self.start_act, self.end_act, self.rl_start, self.rl_end)

    def LoadModel(self):
        self.act_model = self.obj_act_pred.load_model()
        self.dur_model = self.obj_dur_pred.load_model()
        self.start_model = self.obj_start_pred.load_model()

    def _update_log_copy(self, log_copy, predicted_activity_instances):
        # first make their endtimestmap none
        log_copy = log_copy.reset_index(drop=True)
        predicted_activity_instances = predicted_activity_instances.reset_index(drop=True)

        # Ensure consistent data types for 'caseid'
        log_copy['caseid'] = log_copy['caseid'].astype(str)
        predicted_activity_instances['caseid'] = predicted_activity_instances['caseid'].astype(str)

        log_copy = log_copy[~log_copy['caseid'].isin(predicted_activity_instances['caseid'])]
        log_copy = log_copy.append(predicted_activity_instances)
        return log_copy

    def completing_ongoing_activity_instance(self, log_copy, next_start):

        log_copy = log_copy.sort_values(by='start_timestamp')
        log_copy = log_copy.reset_index(drop=True)
        active, activity_instances = feat.get_active_cases(next_start, log_copy)

        activity_instances.loc[activity_instances['end_timestamp'] > next_start, 'end_timestamp'] = None

        activity_instances_incomplete = activity_instances[
            (activity_instances['end_timestamp'] > next_start) | (activity_instances['end_timestamp'].isna())]

        activity_instances_comp = self.obj_dur_pred.trace_completion(activity_instances_incomplete,
                                                                     self.resource_count)

        traces = log_copy[
            (log_copy['caseid'].isin(activity_instances_comp['caseid'])) & (log_copy['start_timestamp'] <= next_start)]

        # Columns to check for matching rows
        comparison_cols = ['caseid', 'ac_index', 'rl_index',
                                               'start_timestamp']

        # Identify common rows based on specific columns
        common_rows = pd.merge(traces, activity_instances_comp, on=comparison_cols)

        # Drop rows in df1 that match with df2 based on the specified columns
        traces = traces[~traces[comparison_cols].apply(tuple, axis=1).isin(common_rows[comparison_cols].apply(tuple, axis=1))]

        traces = traces.append(activity_instances_comp)
        del activity_instances_comp, activity_instances_incomplete, activity_instances

        if not traces.empty:
            # first make their endtimestmap none
            traces['start_timestamp'] = pd.to_datetime(traces['start_timestamp'], errors='coerce')
            traces = traces.sort_values(by='start_timestamp')
            traces.reset_index(drop=True, inplace=True)
            # traces.loc[traces['end_timestamp'] > next_start, 'end_timestamp'] = None
            # print("predicting the next activity instance dur\n")
            predicted_activity_instances = self.obj_dur_pred.predict(traces, self.dur_model)

            # here now predict the next activity instance
            # call next activity instance of incomplete case
            predicted_activity_instances = self.next_activity_instance_of_incomplete_case(predicted_activity_instances)

            log_copy = self._update_log_copy(log_copy, predicted_activity_instances)
            del predicted_activity_instances, traces

        return log_copy

    def next_activity_instance_of_incomplete_case(self, traces):

        traces['start_timestamp'] = pd.to_datetime(traces['start_timestamp'], errors='coerce')
        traces = traces.sort_values(by='start_timestamp')

        # print("predicting the next activity instance\n")
        traces = self.obj_act_pred.predict(traces, self.act_model)
        # print("predicting the next activity instance start\n")
        traces = self.obj_start_pred.predict(traces, self.start_model)

        return traces

    def simulate(self):

        removed_traces = pd.DataFrame()

        # get the dict of cases
        log_copy_ = copy.deepcopy(self.log)
        log_copy_.reset_index(drop=True, inplace=True)

        log_copy_.drop_duplicates(subset=['caseid', 'ac_index', 'rl_index',
                                          'start_timestamp'], inplace=True)
        # duplicates = log_copy[
        #     log_copy.duplicated(subset=['caseid', 'ac_index', 'rl_index', 'start_timestamp'], keep=False)]
        # print(duplicates)

        next_start = self.cut_off
        before_cutoff = log_copy_[log_copy_['start_timestamp'] <= self.cut_off]
        # simulating the sweep line approcah
        log_copy = before_cutoff.copy()
        log_copy['caseid'] = log_copy['caseid'].astype(str)

        while not log_copy.empty and next_start is not None:

            log_copy = self.completing_ongoing_activity_instance(log_copy, next_start)
            log_copy['caseid'] = log_copy['caseid'].astype(str)


            # Iterate through unique traces in the log_copy
            for case_id in log_copy['caseid'].unique():
                # Extract the trace for the given case ID
                trace = log_copy[log_copy['caseid'] == case_id]

                # Check if the trace length exceeds max_len
                if len(trace) >= self.max_len:
                    print(f"Removing trace {case_id} because it exceeds max_len")
                    removed_traces = pd.concat([removed_traces, trace], ignore_index=True)  # Copy to removed traces
                    log_copy = log_copy[log_copy['caseid'] != case_id]  # Remove from log_copy


                # Check if the trace contains an end activity
                if self.end_act in trace['ac_index'].values:  # Adjust 'task' column name as per your DataFrame
                    print(f"Removing trace {case_id} because it has an end activity")
                    removed_traces = pd.concat([removed_traces, trace], ignore_index=True)  # Copy to removed traces
                    log_copy = log_copy[log_copy['caseid'] != case_id]  # Remove from log_copy

            print('current next_start:', next_start)
            print("Status of the log_copy\n")
            print(log_copy[['caseid','start_timestamp','end_timestamp']])
            print("Status of the removed_traces\n")
            if not removed_traces.empty:
                print(removed_traces[['caseid','start_timestamp']])

            # Ensure 'start_timestamp' column is in datetime format
            log_copy['start_timestamp'] = pd.to_datetime(log_copy['start_timestamp'], errors='coerce')

            # Filter valid timestamps (exclude NaT)
            valid_timestamps = log_copy[log_copy['start_timestamp'].notna()]

            # Sort by start_timestamp
            valid_timestamps_log = valid_timestamps.sort_values(by='start_timestamp')

            valid_timestamps = valid_timestamps_log[valid_timestamps_log['start_timestamp'] > next_start]
            next_start = valid_timestamps['start_timestamp'].min() if not valid_timestamps.empty else None


            # # Get the next minimum start timestamp that is greater than the current `next_start`
            # if pd.isna(next_start) or next_start is None:
            #     next_start = valid_timestamps_log['start_timestamp'].min()

            del valid_timestamps_log, valid_timestamps

            # Debugging information
            print("Updated next_start:", next_start)

            gc.collect()



        # when the simulation has ended create the result record

        if log_copy.empty and removed_traces.empty:
            # If both DataFrames are empty, final_results should also be empty
            final_results = pd.DataFrame()
            print("Both log_copy and removed_traces are empty. Final results will also be empty.")
        elif log_copy.empty:
            # If log_copy is empty, final_results should just be the removed_traces
            if 'caseid' in removed_traces.columns:
                final_results = removed_traces.copy()
                print("log_copy is empty. Final results are derived from removed_traces.")
            else:
                # Handle the case where 'caseid' is missing in removed_traces
                final_results = removed_traces.copy()
                print("log_copy is empty, but removed_traces has no 'caseid' column.")
        elif removed_traces.empty:
            # If removed_traces is empty, final_results is just log_copy
            final_results = log_copy.copy()
            print("removed_traces is empty. Final results are derived from log_copy.")
        else:
            # If neither DataFrame is empty, proceed with normal processing
            if 'caseid' in removed_traces.columns and 'caseid' in log_copy.columns:
                # Create the final results DataFrame
                final_results = log_copy.copy()

                # Add missing traces from removed_traces to final_results
                missing_case_ids = set(removed_traces['caseid']) - set(log_copy['caseid'])
                if missing_case_ids:
                    missing_traces = removed_traces[removed_traces['caseid'].isin(missing_case_ids)]
                    final_results = pd.concat([final_results, missing_traces], ignore_index=True)
                else:
                    print("No missing case IDs found. Final results are identical to log_copy.")
            else:
                # Handle cases where 'caseid' is missing in either DataFrame
                print("Warning: 'caseid' column is missing in one or both DataFrames.")
                final_results = log_copy.copy()  # Default to log_copy if caseid-based logic cannot be applied

        # Sort final results for consistent output
        final_results = final_results.sort_values(by=['caseid', 'start_timestamp']).reset_index(drop=True)

        results = self.return_results(final_results, self.log)
        return results

    def return_results(self, log_copy, log):

        # remove rows from log_copy having Endtimestam as none
        log_copy = log_copy[~log_copy['end_timestamp'].isna()]
        log_copy['caseid'] = log_copy['caseid'].astype(str)
        log['caseid'] = log['caseid'].astype(str)

        predictions = self.reformat_events(log_copy, ['ac_index', 'rl_index', 'dur', 'wait'], None)
        ground_truth = self.reformat_events(log, ['ac_index', 'rl_index', 'dur', 'wait'], None)

        results = self.create_result_record(predictions, ground_truth)

        return results

    def create_result_record(self, predictions, ground_truth):
        results = list()

        for case in predictions:
            # get the case with the same caseid in predictions and ground_truth
            caseid = case['caseid']
            ground_truth_case = next((x for x in ground_truth if x['caseid'] == caseid), None)

            if ground_truth_case:
                record = dict()
                record['ac_expect'] = ground_truth_case['ac_index']
                record['ac_pred'] = case['ac_index'] + [self.end_act]  # Concatenate with a list containing end_act
                record['rl_expect'] = ground_truth_case['rl_index']
                record['rl_pred'] = case['rl_index'] + [self.rl_end]  # Concatenate with a list containing rl_end
                record['dur_expect'] = ground_truth_case['dur']
                record['dur_pred'] = case['dur'] + [0]  # Append 0 to the duration list
                record['wait_expect'] = ground_truth_case['wait']
                record['wait_pred'] = case['wait'] + [0]  # Append 0 to the wait list



                results.append(record)

        return results

    def reformat_events(self, log, columns, prefix):
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

                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)
        return temp_data

    @staticmethod
    def rescale(value, parms, scale_args):
        if parms['norm_method'] == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
            # value = np.expm1(np.clip(value, None, 700))  # Clip the value to avoid overflow
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
