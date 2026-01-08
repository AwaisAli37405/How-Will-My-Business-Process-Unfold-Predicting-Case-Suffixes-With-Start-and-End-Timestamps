"""
Created on Fri Jan 10 11:40:46 2020

@author: Awais Ali
For caise23

"""
import random
import itertools
from operator import itemgetter

import jellyfish as jf
import swifter
# from permetrics import RegressionMetric
from scipy.optimize import linear_sum_assignment

from model_prediction.analyzers import alpha_oracle as ao
from model_prediction.analyzers.alpha_oracle import Rel
from minineedle import needle, smith, core
from minineedle.core import Gap

import pandas as pd
import numpy as np
import os

import pickle


class Evaluator():

    def __init__(self, one_timestamp):
        """constructor"""
        self.one_timestamp = one_timestamp

    def measure(self, metric, data, feature=None):
        evaluator = self._get_metric_evaluator(metric)
        return evaluator(data, feature)

    def _get_metric_evaluator(self, metric):
        if metric == 'accuracy':
            return self._accuracy_evaluation
        if metric == 'mae_next':
            return self._mae_next_evaluation
        elif metric == 'similarity':
            return self._similarity_evaluation
        elif metric == 'mae_suffix':
            return self._mae_remaining_evaluation
        elif metric == 'els':
            return self._els_metric_evaluation
        elif metric == 'els_min':
            return self._els_min_evaluation
        elif metric == 'mae_log':
            return self._mae_metric_evaluation
        elif metric == 'dl':
            return self._dl_distance_evaluation
        elif metric == 'jaccard_similarity':
            return self.compute_jaccard_sim
        elif metric == 'RAD':
            return self.RAD
        
        else:
            raise ValueError(metric)

    def _accuracy_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation']]
        eval_acc = (lambda x:
                    1 if x[feature + '_expect'] == x[feature + '_pred'] else 0)
        data[feature + '_acc'] = data.apply(eval_acc, axis=1)
        # agregate true positives
        data = (data.groupby(['implementation', 'run_num'])[feature + '_acc']
                .agg(['sum', 'count'])
                .reset_index())
        # calculate accuracy
        data['accuracy'] = np.divide(data['sum'], data['count'])
        return data

    def _mae_next_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'implementation']]
        ae = (lambda x: np.abs(x[feature + '_expect'] - x[feature + '_pred']))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['implementation', 'run_num'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        return data

    def coverage_ratio(self, original_sequence, predicted_sequence):
        """
        Calculate the coverage ratio of events from the original sequence in the predicted sequence.

        Parameters:
        - original_sequence (list): The original list of events.
        - predicted_sequence (list): The predicted list of events.

        Returns:
        - float: The coverage ratio of events from the original sequence in the predicted sequence.
        """

        # Calculate how many unique events in the original sequence are present in the predicted sequence

        covered_events = None
        try:
            covered_events = sum(1 for event in set(original_sequence) if event in predicted_sequence)
        except TypeError as e:
            # Handle TypeError
            print(f"TypeError: {e}")
        # Calculate the coverage ratio
        ratio = covered_events / len(set(original_sequence))

        return ratio

    def compute_coverage(self, row, feature):
        return self.coverage_ratio(row[feature + '_expect'], row[feature + '_pred'])
    
    def calculate_penalty(self, ground_truth, predicted_suffix):
        # Count occurrences of each activity in ground truth
        ground_truth_counts = {}
        for activity in ground_truth:
            ground_truth_counts[activity] = ground_truth_counts.get(activity, 0) + 1

        # Count occurrences of each activity in predicted suffix
        predicted_counts = {}
        for activity in predicted_suffix:
            predicted_counts[activity] = predicted_counts.get(activity, 0) + 1

        # Calculate penalties
        total_penalty = 0
        all_activities = set(ground_truth) | set(predicted_suffix)
        for activity in all_activities:
            ground_truth_count = ground_truth_counts.get(activity, 0)
            predicted_count = predicted_counts.get(activity, 0)
            total_penalty += abs(ground_truth_count - predicted_count)

        return total_penalty
    
    def RAD(self, log, parms):

        data = log.copy()
        data = data[
            ['pref_size','ac_expect', 'ac_pred']]

        print(data)

        data = data.to_dict('records')

        total_penalty = 0
        total_max_possible_penalty = 0

        for entry in data:
            ground_truth = entry['ac_expect']
            predicted_suffix = entry['ac_pred']

            penalty = self.calculate_penalty(ground_truth, predicted_suffix)
            total_penalty += penalty

            # max_possible_penalty = len(set(ground_truth) | set(predicted_suffix))
            max_possible_penalty =  len(ground_truth) + len(predicted_suffix)
            total_max_possible_penalty += max_possible_penalty

        # Normalize the total penalty
        normalized_total_penalty = 0
        if total_max_possible_penalty != 0:
            normalized_total_penalty = total_penalty / total_max_possible_penalty

        print(f"Normalized Total Penalty: {normalized_total_penalty}")

        return normalized_total_penalty
    
    
    
    

    def compute_jaccard_sim(self, data, parms):
        def correct_union(intervals_union):
            if (len(intervals_union) != 0):
                intervals_union = sorted(set(intervals_union), key=lambda x: x[0])
                union_intervals = 0
                start, end = intervals_union[0]
        
                for interval in intervals_union:
                    if interval[0] > end:
                        union_intervals += end - start
                        start, end = interval
                    else:
                        end = max(end, interval[1])
        
                #union_intervals += end - start
                union_intervals = (start,end)
                return union_intervals
            else:
                return (0,0)
                #return 0

        def cal_union(intervals):
            if (len(intervals) != 0):
                intervals_union = sorted(set(intervals), key=lambda x: x[0])
                union_intervals = 0
                start, end = intervals_union[0]
    
                for interval in intervals_union:
                    if interval[0] > end:
                        union_intervals += end - start
                        start, end = interval
                    else:
                        end = max(end, interval[1])
    
                union_intervals += end - start
    
                return [union_intervals]
            else:
                return [0]

        
        
        data = data.copy()
        data = data[
            ['pref_size', 'ac_expect', 'ac_pred', 'dur_expect', 'dur_pred', 'wait_expect', 'wait_pred']]

        print(data)

        jaccard_similarity_dur = []
        multiset_dur = []

        jaccard_similarity_wait = []
        multiset_wait = []

        log = data.to_dict('records')
        
        multiset_intersection_joint = []
        multiset_union_joint = []
        
        multiset_intersection_dur = []
        multiset_union_dur = []
        
        multiset_intersection_wait = []
        multiset_union_wait = []
        
        coalesce_intersection_joint = []
        coalesce_union_joint = []
        
        coalesce_intersection_dur = []
        coalesce_union_dur = []
        
        coalesce_intersection_wait = []
        coalesce_union_wait = []
        
        
        for record in log:

            dur_pred = [max(item, 0) for item in record['dur_pred']]  # replacing the neg with zero
            dur_expect = [item for item in record['dur_expect']]
            wait_pred = [max(item, 0) for item in record['wait_pred']]
            wait_expect = [item for item in record['wait_expect']]

            def create_intervals_from_durations(durations, wait, start_ref=0):
                
                intervals_joint = []
                current_start = None
                
                for i in range(len(durations)):
                    if i==0:
                        current_start = start_ref + wait[i]
                    
                    end = current_start + durations[i]
                    intervals_joint.append((current_start, end))
                    wait_value = wait[i + 1] if i + 1 < len(wait) else end
                    current_start += wait_value
                    
                #return intervals
                duration_intervals = []
                wait_intervals = []
                dur_start = 0
                wait_start = 0
                # for duration,wait_ in zip(durations,wait):
                #     end = current_start + duration
                #     intervals.append((current_start, end))
                #     # current_start = duration + (wait_ - duration)
                #     current_start +=wait_

                for i in range(len(durations)):
                    duration = durations[i]

                    end = dur_start + duration
                    duration_intervals.append((dur_start, end))
                    dur_start = end
                    # wait_value = wait[i + 1] if i + 1 < len(wait) else end
                    # current_start += wait_value
                    start = wait_start + wait[i]
                    wait_intervals.append((wait_start, start))
                    wait_start = start

                return duration_intervals, wait_intervals, intervals_joint

            def coalesce_intervals(intervals):
                if not intervals:
                    return []

                # Sort the intervals by their start times
                intervals.sort(key=lambda x: x[0])

                # Start with the first interval
                merged = [intervals[0]]

                for current_start, current_end in intervals:
                    # If the current interval overlaps with the previous interval,
                    # merge them by updating the end time of the previous interval.
                    if current_start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], current_end))
                    else:
                        # Otherwise, add the current interval as is
                        merged.append((current_start, current_end))

                return merged

            def find_all_intersections_jaccard_sim(chosen_interval, intervals_list, used_intervals_set):
                """Return a list of all intervals in intervals_list that intersect with chosen_interval."""
                a1, a2 = chosen_interval
                intersections = []
                sum_intersections = 0
                intervals_union_ = []
                union_intervals = 0
                for b1, b2 in intervals_list:
                    if a1 <= b2 and b1 <= a2:  # The intervals intersect
                        intersection = (max(a1, b1), min(a2, b2))
                        intersections.append(intersection)
    
                        # Sum of intersections
                        x, y = intersection
                        inter_value = (y - x)
                        sum_intersections += inter_value
                        
                        intervals_union_.append((b1, b2))
                        
                        intervals_union_.append(chosen_interval)
                        if (len(intervals_union_) != 0):
                            intervals_union = sorted(set(intervals_union_), key=lambda x: x[0])
                            union_intervals = 0
                            start, end = intervals_union[0]
                
                            for interval in intervals_union:
                                if interval[0] > end:
                                    union_intervals += end - start
                                    start, end = interval
                                else:
                                    end = max(end, interval[1])
                
                            union_ = end - start
                            union_ = union_ - inter_value
                            union_intervals += union_
                               
                return sum_intersections, union_intervals, used_intervals_set
                
                
            # intervals have been created
            duration_intervals_pred, wait_intervals_pred, intervals_joint_pred = create_intervals_from_durations(dur_pred, wait_pred)
            duration_intervals_actual, wait_intervals_actual, intervals_joint_actual = create_intervals_from_durations(dur_expect, wait_expect)
            
            joint_coalesce_predicted = coalesce_intervals(intervals_joint_pred)
            joint_coalesce_actual = coalesce_intervals(intervals_joint_actual)
            
            coalesce_predicted_dur = coalesce_intervals(duration_intervals_pred)
            coalesce_actual_dur = coalesce_intervals(duration_intervals_actual)
            
            coalesce_predicted_wait = coalesce_intervals(wait_intervals_pred)
            coalesce_actual_wait = coalesce_intervals(wait_intervals_actual)
            ###############################################################################
            used_intervals_set = set()
            # finding the intersections for coalesce intervals
            for int_ in joint_coalesce_predicted:
                intersections, union, used_intervals_set = find_all_intersections_jaccard_sim(int_, joint_coalesce_actual,used_intervals_set)
                coalesce_intersection_joint.append(intersections)
                coalesce_union_joint.append(union)
            # union_ = correct_union(used_intervals_set)
            # coalesce_union_joint.append(union_)
    
            used_intervals_set =set()
            for int_ in coalesce_predicted_dur:
                intersections, union, used_intervals_set = find_all_intersections_jaccard_sim(int_, coalesce_actual_dur,used_intervals_set)
                coalesce_intersection_dur.append(intersections)
                coalesce_union_dur.append(union)
            # union_ = correct_union(used_intervals_set)
            # coalesce_union_dur.append(union_)
    
            used_intervals_set = set()
            for int_ in coalesce_predicted_wait:
                intersections, union, used_intervals_set = find_all_intersections_jaccard_sim(int_, coalesce_actual_wait,used_intervals_set)
                coalesce_intersection_wait.append(intersections)
                coalesce_union_wait.append(union)
            # union_ = correct_union(used_intervals_set)
            # coalesce_union_wait.append(union_)
            # ################################################################################
    
            # finding the intersection for regular intervals
            used_intervals_set =set()
            for int_ in intervals_joint_pred:
                intersections, union, used_intervals_set = find_all_intersections_jaccard_sim(int_, intervals_joint_actual,used_intervals_set)
                multiset_intersection_joint.append(intersections)
                multiset_union_joint.append(union)
            # union_ = correct_union(used_intervals_set)
            # multiset_union_joint.append(union_)
    
            used_intervals_set = set()
            for int_ in duration_intervals_pred:
                intersections, union, used_intervals_set = find_all_intersections_jaccard_sim(int_, duration_intervals_actual,used_intervals_set)
                multiset_intersection_dur.append(intersections)
                multiset_union_dur.append(union)
            # union_ = correct_union(used_intervals_set)
            # multiset_union_dur.append(union_)
    
    
            used_intervals_set =set()
            for int_ in wait_intervals_pred:
                intersections, union, used_intervals_set = find_all_intersections_jaccard_sim(int_, wait_intervals_actual,used_intervals_set)
                multiset_intersection_wait.append(intersections)
                multiset_union_wait.append(union)
            # union_ = correct_union(used_intervals_set)
            # multiset_union_wait.append(union_)
    
        sum_inter = sum(coalesce_intersection_joint)
        # coalesce_union_joint = cal_union(coalesce_union_joint)
        sum_union = sum(coalesce_union_joint)
    
        c_joint = sum_inter / sum_union
    
        sum_inter = sum(coalesce_intersection_dur)
        # coalesce_union_dur = cal_union(coalesce_union_dur)
        sum_union = sum(coalesce_union_dur)
    
        c_dur = sum_inter / sum_union
    
        sum_inter = sum(coalesce_intersection_wait)
        # coalesce_union_wait = cal_union(coalesce_union_wait)
        sum_union = sum(coalesce_union_wait)
    
        c_wait = sum_inter / sum_union
    
        sum_inter = sum(multiset_intersection_joint)
        # multiset_union_joint = cal_union(multiset_union_joint)
        sum_union = sum(multiset_union_joint)
    
        m_joint = sum_inter / sum_union
    
        sum_inter = sum(multiset_intersection_dur)
        # multiset_union_dur = cal_union(multiset_union_dur)
        sum_union = sum(multiset_union_dur)
    
        m_dur = sum_inter / sum_union
    
        sum_inter = sum(multiset_intersection_wait)
        # multiset_union_wait = cal_union(multiset_union_wait)
        sum_union = sum(multiset_union_wait)
    
        m_wait = sum_inter / sum_union
    
        # Printing the results before returning
        print(f"c_joint (Coalesced Joint Jaccard Similarity): {c_joint}")
        print(f"c_dur (Coalesced Duration Jaccard Similarity): {c_dur}")
        print(f"c_wait (Coalesced Wait Jaccard Similarity): {c_wait}")
        print(f"m_joint (Multiset Joint Jaccard Similarity): {m_joint}")
        print(f"m_dur (Multiset Duration Jaccard Similarity): {m_dur}")
        print(f"m_wait (Multiset wait Jaccard Similarity): {m_wait}")
    
        return c_joint, c_dur, c_wait, m_joint, m_dur, m_wait
    

    def _similarity_evaluation(self, data, feature):
        data = data.copy()
        # data = data[[(feature + '_expect'), (feature + '_pred'),
        #              'run_num', 'implementation', 'pref_size']]
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'pref_size']]

        data['coverage'] = data.apply(self.compute_coverage, axis=1, args=(feature,))

        # append all values and create alias
        values = (data[feature + '_pred'].tolist() +
                  data[feature + '_expect'].tolist())
        values = list(set(itertools.chain.from_iterable(values)))
        index = self.create_task_alias(values)
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

        # agregate similarities
        # data = (data.groupby(['implementation', 'run_num', 'pref_size'])['similarity']
        # .agg(['mean'])
        # .reset_index()
        # .rename(columns={'mean': 'similarity'}))
        # data = (data.groupby(['pref_size'])['similarity']
        #         .agg(['mean'])
        #         .reset_index()
        #         .rename(columns={'mean': 'similarity'}))

        data = (data.groupby(['pref_size'])
                .agg({'similarity': 'mean', 'coverage': 'mean'})
                .reset_index()
                .rename(columns={'similarity': 'similarity', 'coverage': 'coverage'}))

        # data = (pd.pivot_table(data,
        #                        values='similarity',
        #                        # index=['run_num', 'implementation'],
        #                        columns=['pref_size'],
        #                        aggfunc=np.mean,
        #                        fill_value=0,
        #                        margins=True,
        #                        margins_name='mean')
        #         .reset_index())
        # data = data[data.run_num != 'mean']
        return data

    def _mae_remaining_evaluation(self, data, feature):
        data = data.copy()
        # data = data[[(feature + '_expect'), (feature + '_pred'),
        #              'run_num', 'implementation', 'pref_size']]
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'pref_size']]
        ae = (lambda x: np.abs(np.sum(x[feature + '_expect']) -
                               np.sum(x[feature + '_pred'])))
        data['ae'] = data.apply(ae, axis=1)
        # data = (data.groupby(['implementation', 'run_num', 'pref_size'])['ae']
        #         .agg(['mean'])
        #         .reset_index()
        #         .rename(columns={'mean': 'mae'}))
        data = (data.groupby(['pref_size'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        # data = (pd.pivot_table(data,
        #                        values='mae',
        #                        # index=['run_num', 'implementation'],
        #                        columns=['pref_size'],
        #                        aggfunc=np.mean,
        #                        fill_value=0,
        #                        margins=True,
        #                        margins_name='mean')
        #         .reset_index())
        # data = data[data.run_num != 'mean']
        return data

    # =============================================================================
    # Timed string distance
    # =============================================================================
    def _els_metric_evaluation(self, data, feature):
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation != 'log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            mx_len = len(log_data)
            cost_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
            # Create cost matrix
            # start = timer()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    comp_sec = self.create_comparison_elements(pred_data,
                                                               log_data, i, j)
                    length = np.max([len(comp_sec['seqs']['s_1']),
                                     len(comp_sec['seqs']['s_2'])])
                    distance = self.tsd_alpha(comp_sec,
                                              alpha_concurrency.oracle) / length
                    cost_matrix[i][j] = distance
            # end = timer()
            # print(end - start)
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                       sim_order=pred_data[idx]['profile'],
                                       log_order=log_data[idy]['profile'],
                                       sim_score=(1 - (cost_matrix[idx][idy])),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'els'}))
        return data

    def _els_min_evaluation(self, data, feature):
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation != 'log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            temp_log_data = log_data.copy()
            for i in range(0, len(pred_data)):
                comp_sec = self.create_comparison_elements(pred_data,
                                                           temp_log_data, i, 0)
                min_dist = self.tsd_alpha(comp_sec, alpha_concurrency.oracle)
                min_idx = 0
                for j in range(1, len(temp_log_data)):
                    comp_sec = self.create_comparison_elements(pred_data,
                                                               temp_log_data, i, j)
                    sim = self.tsd_alpha(comp_sec, alpha_concurrency.oracle)
                    if min_dist > sim:
                        min_dist = sim
                        min_idx = j
                length = np.max([len(pred_data[i]['profile']),
                                 len(temp_log_data[min_idx]['profile'])])
                similarity.append(dict(caseid=pred_data[i]['caseid'],
                                       sim_order=pred_data[i]['profile'],
                                       log_order=temp_log_data[min_idx]['profile'],
                                       sim_score=(1 - (min_dist / length)),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
                del temp_log_data[min_idx]
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'els'}))
        return data

    def create_comparison_elements(self, serie1, serie2, id1, id2):
        """
        Creates a dictionary of the elements to compare

        Parameters
        ----------
        serie1 : List
        serie2 : List
        id1 : integer
        id2 : integer

        Returns
        -------
        comp_sec : dictionary of comparison elements

        """
        comp_sec = dict()
        comp_sec['seqs'] = dict()
        comp_sec['seqs']['s_1'] = serie1[id1]['profile']
        comp_sec['seqs']['s_2'] = serie2[id2]['profile']
        comp_sec['times'] = dict()
        if self.one_timestamp:
            comp_sec['times']['p_1'] = serie1[id1]['dur_act_norm']
            comp_sec['times']['p_2'] = serie2[id2]['dur_act_norm']
        else:
            comp_sec['times']['p_1'] = serie1[id1]['dur_act_norm']
            comp_sec['times']['p_2'] = serie2[id2]['dur_act_norm']
            comp_sec['times']['w_1'] = serie1[id1]['wait_act_norm']
            comp_sec['times']['w_2'] = serie2[id2]['wait_act_norm']
        return comp_sec

    def tsd_alpha(self, comp_sec, alpha_concurrency):
        """
        Compute the Damerau-Levenshtein distance between two given
        strings (s_1 and s_2)
        Parameters
        ----------
        comp_sec : dict
        alpha_concurrency : dict
        Returns
        -------
        Float
        """
        s_1 = comp_sec['seqs']['s_1']
        s_2 = comp_sec['seqs']['s_2']
        dist = {}
        lenstr1 = len(s_1)
        lenstr2 = len(s_2)
        for i in range(-1, lenstr1 + 1):
            dist[(i, -1)] = i + 1
        for j in range(-1, lenstr2 + 1):
            dist[(-1, j)] = j + 1
        for i in range(0, lenstr1):
            for j in range(0, lenstr2):
                if s_1[i] == s_2[j]:
                    cost = self.calculate_cost(comp_sec['times'], i, j)
                else:
                    cost = 1
                dist[(i, j)] = min(
                    dist[(i - 1, j)] + 1,  # deletion
                    dist[(i, j - 1)] + 1,  # insertion
                    dist[(i - 1, j - 1)] + cost  # substitution
                )
                if i and j and s_1[i] == s_2[j - 1] and s_1[i - 1] == s_2[j]:
                    if alpha_concurrency[(s_1[i], s_2[j])] == Rel.PARALLEL:
                        cost = self.calculate_cost(comp_sec['times'], i, j - 1)
                    dist[(i, j)] = min(dist[(i, j)], dist[i - 2, j - 2] + cost)  # transposition
        return dist[lenstr1 - 1, lenstr2 - 1]

    def calculate_cost(self, times, s1_idx, s2_idx):
        """
        Takes two events and calculates the penalization based on mae distance

        Parameters
        ----------
        times : dict with lists of times
        s1_idx : integer
        s2_idx : integer

        Returns
        -------
        cost : float
        """
        if self.one_timestamp:
            p_1 = times['p_1']
            p_2 = times['p_2']
            cost = np.abs(p_2[s2_idx] - p_1[s1_idx]) if p_1[s1_idx] > 0 else 0
        else:
            p_1 = times['p_1']
            p_2 = times['p_2']
            w_1 = times['w_1']
            w_2 = times['w_2']
            t_1 = p_1[s1_idx] + w_1[s1_idx]
            if t_1 > 0:
                b_1 = (p_1[s1_idx] / t_1)
                cost = ((b_1 * np.abs(p_2[s2_idx] - p_1[s1_idx])) +
                        ((1 - b_1) * np.abs(w_2[s2_idx] - w_1[s1_idx])))
            else:
                cost = 0
        return cost

    # =============================================================================
    # dl distance
    # =============================================================================
    def _dl_distance_evaluation(self, data, feature):
        """
        similarity score

        Parameters
        ----------
        log_data : list of events
        simulation_data : list simulation event log

        Returns
        -------
        similarity : float

        """
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        # alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation != 'log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            mx_len = len(log_data)
            dl_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
            # Create cost matrix
            # start = timer()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    d_l = self.calculate_distances(pred_data, log_data, i, j)
                    dl_matrix[i][j] = d_l
            # end = timer()
            # print(end - start)
            dl_matrix = np.array(dl_matrix)
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(dl_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                       sim_order=pred_data[idx]['profile'],
                                       log_order=log_data[idy]['profile'],
                                       sim_score=(1 - (dl_matrix[idx][idy])),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'dl'}))
        return data

    @staticmethod
    def calculate_distances(serie1, serie2, id1, id2):
        """
        Parameters
        ----------
        serie1 : list
        serie2 : list
        id1 : index of the list 1
        id2 : index of the list 2

        Returns
        -------
        dl : float value
        ae : absolute error value
        """
        length = np.max([len(serie1[id1]['profile']),
                         len(serie2[id2]['profile'])])
        d_l = jf.damerau_levenshtein_distance(
            ''.join(serie1[id1]['profile']),
            ''.join(serie2[id2]['profile'])) / length
        return d_l

    # =============================================================================
    # mae distance
    # =============================================================================

    def _mae_metric_evaluation(self, data, feature):
        """
        mae distance between logs

        Parameters
        ----------
        log_data : list of events
        simulation_data : list simulation event log

        Returns
        -------
        similarity : float

        """
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.implementation == 'log']
        alias = self.create_task_alias(data.task.unique())
        # alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'implementation']].drop_duplicates()
        variants = variants[variants.implementation != 'log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.implementation == var['implementation']) &
                             (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                             'task',
                                             alias)
            mx_len = len(log_data)
            ae_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
            # Create cost matrix
            # start = timer()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    cicle_time_s1 = (pred_data[i]['end_time'] -
                                     pred_data[i]['start_time']).total_seconds()
                    cicle_time_s2 = (log_data[j]['end_time'] -
                                     log_data[j]['start_time']).total_seconds()
                    ae = np.abs(cicle_time_s1 - cicle_time_s2)
                    ae_matrix[i][j] = ae
            # end = timer()
            # print(end - start)
            ae_matrix = np.array(ae_matrix)
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(ae_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                       sim_order=pred_data[idx]['profile'],
                                       log_order=log_data[idy]['profile'],
                                       sim_score=(ae_matrix[idx][idy]),
                                       implementation=var['implementation'],
                                       run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['implementation', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae_log'}))
        return data

    # =============================================================================
    # Support methods
    # =============================================================================
    @staticmethod
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

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['duration'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instant since there is no previous timestamp
                if self.one_timestamp:
                    if i == 0:
                        dur = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i - 1]['end_timestamp']).total_seconds()
                else:
                    dur = (events[i]['end_timestamp'] -
                           events[i]['start_timestamp']).total_seconds()
                    if i == 0:
                        wit = 0
                    else:
                        wit = (events[i]['start_timestamp'] -
                               events[i - 1]['end_timestamp']).total_seconds()
                    events[i]['waiting'] = wit
                events[i]['duration'] = dur
        return pd.DataFrame.from_dict(log)

    def scaling_data(self, data):
        """
        Scales times values activity based

        Parameters
        ----------
        data : dataframe

        Returns
        -------
        data : dataframe with normalized times

        """
        df_modif = data.copy()
        np.seterr(divide='ignore')
        summ = data.groupby(['task'])['duration'].max().to_dict()
        dur_act_norm = (lambda x: x['duration'] / summ[x['task']]
        if summ[x['task']] > 0 else 0)
        df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        if not self.one_timestamp:
            summ = data.groupby(['task'])['waiting'].max().to_dict()
            wait_act_norm = (lambda x: x['waiting'] / summ[x['task']]
            if summ[x['task']] > 0 else 0)
            df_modif['wait_act_norm'] = df_modif.apply(wait_act_norm, axis=1)
        return df_modif

    def reformat_events(self, data, features, alias):
        """Creates series of activities, roles and relative times per trace.
        parms:
            log_df: dataframe.
            ac_table (dict): index of activities.
            rl_table (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        # Update alias
        if isinstance(features, list):
            [x.update(dict(alias=alias[(x[features[0]],
                                        x[features[1]])])) for x in data]
        else:
            [x.update(dict(alias=alias[x[features]])) for x in data]
        temp_data = list()
        # define ordering keys and columns
        if self.one_timestamp:
            columns = ['alias', 'duration', 'dur_act_norm']
            sort_key = 'end_timestamp'
        else:
            sort_key = 'start_timestamp'
            columns = ['alias', 'duration',
                       'dur_act_norm', 'waiting', 'wait_act_norm']
        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {**{'profile': serie}, **temp_dict}
                else:
                    serie = [y[col] for y in trace]
                temp_dict = {**{col: serie}, **temp_dict}
            temp_dict = {**{'caseid': key, 'start_time': trace[0][sort_key],
                            'end_time': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('start_time'))
