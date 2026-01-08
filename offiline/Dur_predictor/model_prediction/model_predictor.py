# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo
"""

import os
import json
import copy
from datetime import timedelta
import pandas as pd
import numpy as np
import configparser as cp

import readers.log_reader as lr
import utils.support as sup

from tensorflow.keras.models import load_model

from model_training import features_manager as feat
from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev

from model_prediction import simulation as sim
from model_prediction import evaluation_utils as eval_utils


class ModelPredictor():
    """
    This is the man class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = os.path.join('output_files', parms['folder'])
        self.parms = parms
        # load parameters
        self.load_parameters()
        self.model_name = os.path.join(self.output_route, parms['model_file'])
        self.model =load_model(os.path.join(self.output_route,
                                             parms['model_file']))
        self.log = self.load_log_test(self.output_route, self.parms)
        # self.log = self.log.iloc[:500]


        self.samples = dict()
        self.predictions = None
        self.sim_values = list()
        self.metrics = None
        self.model_def = dict()
        self.read_model_definition(self.parms['model_type'])
        self.parms['additional_columns'] = self.model_def['additional_columns']
        if self.parms['activity'] == 'sim_start':
            self.acc = self.execute_simulation()
        else:
            self.acc = self.execute_predictive_task()

    def execute_simulation(self):

        # # enrich log
        feat_mannager = feat.FeaturesMannager(self.parms)
        feat_mannager.register_scaler(self.parms['model_type'],
                                      self.model_def['vectorizer'])
        self.log, _ = feat_mannager.calculate(self.log, self.parms['additional_columns'])

        simulation = sim.Simulation(self)
        predictions = simulation.simulate()
        # export predictions
        self.predictions = pd.DataFrame(predictions)
        self.export_predictions(0)
        # evalaute_simulation = self.evalaute_simulation(predictions)
        dl_accuracy = eval_utils._similarity_evaluation(self.predictions, 'ac')
        rl_accuracy = eval_utils._similarity_evaluation(self.predictions, 'rl')
        dur_mae = eval_utils._mae_remaining_evaluation(self.predictions, 'dur')
        wait_mae = eval_utils._mae_remaining_evaluation(self.predictions, 'wait')

        self.metrics = pd.DataFrame([{'Event_log': self.parms['model_file'], 'dl_accuracy': dl_accuracy,
                                      'rl_accuracy': rl_accuracy, 'dur_mae': dur_mae,
                                      'wait_mae': wait_mae}])

        # self.metrics = pd.DataFrame([{'Event_log': self.parms['model_file'], 'dur_mae': evalaute_simulation[0],
        #                               'wait_mae': evalaute_simulation[1]}])
        self._export_results(self.output_route)




    def execute_predictive_task(self):
        # create examples for next event and suffix
        if self.parms['activity'] == 'pred_log':
            self.parms['num_cases'] = len(self.log.caseid.unique())
            self.parms['start_time'] = self.log.start_timestamp.min()
        else:
            feat_mannager = feat.FeaturesMannager(self.parms)
            feat_mannager.register_scaler(self.parms['model_type'],
                                          self.model_def['vectorizer'])
            self.log, _ = feat_mannager.calculate(
                self.log, self.parms['additional_columns'])
            sampler = it.SamplesCreator()
            sampler.create(self, self.parms['activity'])
        # predict
        self.imp = self.parms['variant']
        for run_num in range(0, self.parms['rep']):
            self.predict_values(run_num)
            # export predictions
            self.export_predictions(run_num)
            # assesment
            evaluator = EvaluateTask()
            if self.parms['activity'] == 'pred_log':
                self.sim_values.extend(
                    evaluator.evaluate(self.parms,
                                       self.log,
                                    self.predictions,
                                    run_num))
            else:
                # self.metrics = evaluator.evaluate(self.predictions, self.parms,run_num)
                self.metrics = evaluator.evaluate(self.parms, self.log, self.predictions, run_num)

        self._export_results(self.output_route)

    def predict_values(self, run_num):
        # Predict values
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, self.parms['activity'], run_num)

    @staticmethod
    def load_log_test(output_route, parms):
        df_test = lr.LogReader(
            os.path.join(output_route, 'parameters', 'test_log.csv'),
            parms['read_options'])
        df_test = pd.DataFrame(df_test.data)
        df_test = df_test[~df_test.task.isin(['Start', 'End'])]
        return df_test

    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.output_route,
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            parms = {k: v for k, v in data.items()}
            parms.pop('rep', None)
            self.parms = {**self.parms, **parms}
            if 'dim' in data.keys():
                self.parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
            if self.parms['one_timestamp']:
                self.parms['scale_args'] = {
                    k: float(v) for k, v in data['scale_args'].items()}
            else:
                for key in data['scale_args'].keys():
                    self.parms['scale_args'][key] = {
                        k: float(v) for k, v in data['scale_args'][key].items()}
            self.parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            self.parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
            file.close()
            self.ac_index = {v: k for k, v in self.parms['index_ac'].items()}
            self.rl_index = {v: k for k, v in self.parms['index_rl'].items()}

    def sampling(self, sampler):
        sampler.register_sampler(self.parms['model_type'],
                                 self.model_def['vectorizer'])
        self.samples = sampler.create_samples(
            self.parms, self.log, self.ac_index,
            self.rl_index, self.model_def['additional_columns'])


    def predict(self, executioner, run_num):
        
        results = executioner.predict(self.parms,
                                      self.model,
                                      self.samples,
                                      self.imp,
                                      self.model_def['vectorizer'])
        results = pd.DataFrame(results)
        self.predictions = results

    def export_predictions(self, r_num):
        # output_folder = os.path.join(self.output_route, 'results')
        if not os.path.exists(self.output_route):
            os.makedirs(self.output_route)
        self.predictions.to_csv(
            os.path.join(
                self.output_route, 'gen_' + self.parms['activity'] + '_ ' +
                                   self.parms['model_file'].split('.')[0] + '_' + str(r_num + 1) + '.csv'),
            index=False)

    @staticmethod
    def scale_feature(log, feature, parms, replace=False):
        """Scales a number given a technique.
        Args:
            log: Event-log to be scaled.
            feature: Feature to be scaled.
            method: Scaling method max, lognorm, normal, per activity.
            replace (optional): replace the original value or keep both.
        Returns:
            Scaleded value between 0 and 1.
        """
        method = parms['norm_method']
        scale_args = parms['scale_args']
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
        elif method is None:
            log[feature+'_norm'] = log[feature]
        else:
            raise ValueError(method)
        if replace:
            log = log.drop(feature, axis=1)
        return log

    def read_model_definition(self, model_type):
        Config = cp.ConfigParser(interpolation=None)
        Config.read('models_spec.ini')
        #File name with extension
        self.model_def['additional_columns'] = sup.reduce_list(
            Config.get(model_type,'additional_columns'), dtype='str')
        self.model_def['vectorizer'] = Config.get(model_type, 'vectorizer')

    def _export_results(self, output_path) -> None:
        # Save results
        # pd.DataFrame(self.sim_values).to_csv(
        #     os.path.join(self.output_route, sup.file_id(prefix='SE_')),
        #     index=False)
        # Save logs
        log_test = self.log[~self.log.task.isin(['Start', 'End'])]
        log_test.to_csv(
            os.path.join(self.output_route, 'tst_' +
                         self.parms['model_file'].split('.')[0] + '.csv'),
            index=False)
        self.metrics.to_csv(
            os.path.join(self.output_route, 'result_' + self.parms['activity'] + '_' + self.parms['variant'] + '_' +
                         self.parms['model_file'].split('.')[0] + '.csv'),
            index=False)
        
class EvaluateTask():

    def evaluate(self, parms, log, predictions, rep_num):
        sampler = self._get_evaluator(parms['activity'])
        return sampler(predictions, parms, rep_num)

    # def evaluate(self,data, parms,run):
    #     sampler = self._get_evaluator(parms['activity'])
    #     return sampler(data, parms,run)

    def _get_evaluator(self, activity):
        if activity == 'predict_next':
            return self._evaluate_predict_next
        elif activity == 'pred_sfx':
            return self._evaluate_pred_sfx
        elif activity == 'pred_log':
            return self._evaluate_predict_log
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parms, rep_num):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['one_timestamp'], parms)
        ac_sim = evaluator.measure('accuracy', data, 'ac')
        rl_sim = evaluator.measure('accuracy', data, 'rl')
        # mean_ac = ac_sim.accuracy.mean()
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc] * len(ac_sim), ignore_index=True)
        exp_desc['mean_ac'] = ac_sim['accuracy']
        exp_desc['mean_rl'] = rl_sim['accuracy']

        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_next', data, 'tm')
            tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(tm_mae, 'tm', parms)
        else:
            dur_mae = evaluator.measure('mae_next', data, 'dur')
            wait_mae = evaluator.measure('mae_next', data, 'wait')
            exp_desc['dur_mae'] = dur_mae['mae']
            exp_desc['wait_mea'] = wait_mae['mae']
            dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
            wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(dur_mae, 'dur', parms)
            self.save_results(wait_mae, 'wait', parms)

        return exp_desc

    def _evaluate_pred_sfx(self, data, parms, rep_num):
        exp_desc = self.clean_parameters(parms.copy())
        #import pdb; pdb.set_trace()
        evaluator = ev.Evaluator(parms['one_timestamp'])
        ac_sim = evaluator.measure('similarity', data, 'ac')
        rl_sim = evaluator.measure('similarity', data, 'rl')
        # mean_sim = (ac_sim['mean'].mean())
        
        ##########################  Alignments ##############################
        #jaccard_dur , multiset_dur, jaccard_wait, multiset_wait = evaluator.measure('jaccard_similarity', data, parms)
        rad = evaluator.measure('RAD', data, parms)
        
        #c_joint,c_dur,c_wait,m_joint,m_dur,m_wait = evaluator.measure('jaccard_similarity', data, parms)


        ##############################################################################

        # mean_sim = (ac_sim['mean'].mean())


        exp_desc = pd.DataFrame([exp_desc])
        #exp_desc['c_joint'] = c_joint
        #exp_desc['c_dur'] = c_dur
        #exp_desc['c_wait'] = c_wait
        #exp_desc['m_joint'] = m_joint
        #exp_desc['m_dur'] = m_dur
        #exp_desc['m_wait'] = m_wait
        
        #exp_desc['jaccard_sim_dur'] = sum(jaccard_dur)/len(jaccard_dur)
        #exp_desc['multiset_sim_dur'] = sum(multiset_dur) / len(multiset_dur)
        #exp_desc['jaccard_sim_wait'] = sum(jaccard_wait) / len(jaccard_wait)
        #exp_desc['multiset_sim_wait'] = sum(multiset_wait) / len(multiset_wait)

        #exp_desc = pd.DataFrame([exp_desc])
        # exp_desc = pd.concat([exp_desc]*len(ac_sim), ignore_index=True)
        # ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        # rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')


        mean_sim = ac_sim['similarity'].mean()
        mean_coverage = ac_sim['coverage'].mean()

        mean_rl = rl_sim['similarity'].mean()
        # exp_desc_ac = pd.concat([exp_desc] * 1, ignore_index=True)
        # exp_desc_rl = pd.concat([exp_desc] * 1, ignore_index=True)

        exp_desc['mean_ac_similarity'] = mean_sim
        exp_desc['mean_ac_coverage'] = mean_coverage
        exp_desc['RAD'] = rad
        exp_desc['mean_rl_similarity'] = mean_rl

        # ac_sim = exp_desc_ac.to_dict('records')
        # rl_sim = exp_desc_rl.to_dict('records')

        # self.save_results(ac_sim, 'ac', parms)
        # self.save_results(rl_sim, 'rl', parms)
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_suffix', data, 'tm')
            exp_desc['tm_mae'] = tm_mae['mae'].mean()

            # tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
            # self.save_results(tm_mae, 'tm', parms)
        else:
            dur_mae = evaluator.measure('mae_suffix', data, 'dur')
            wait_mae = evaluator.measure('mae_suffix', data, 'wait')
            exp_desc['dur_mae'] = dur_mae['mae'].mean()
            exp_desc['wait_mae'] = wait_mae['mae'].mean()
            exp_desc['remaining_time_mae'] = exp_desc['dur_mae'] + exp_desc['wait_mae']
            days, hours, minutes, remaining_seconds = self.seconds_to_days(exp_desc['remaining_time_mae'][0])


            # rt = exp_desc['remaining_time_mae'][0] * 24 * 60 * 60
            # duration = pendulum.duration(seconds=rt)
            print(f"{days} days, {hours} hours, {minutes} minutes, and {remaining_seconds} seconds")
            exp_desc['remaining_time_mae_days'] = days
            # dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
            # wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
            # self.save_results(dur_mae, 'dur', parms)
            # self.save_results(wait_mae, 'wait', parms)


        # result = exp_desc.to_dict('records')
        # self.save_results(result, 'suffix_prediction_results', parms)



        return exp_desc

    @staticmethod
    def _evaluate_predict_log(parms, log, sim_log, rep_num):
        """Reads the simulation results stats
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        sim_values = list()
        log = copy.deepcopy(log)
        log = log[~log.task.isin(['Start', 'End', 'start', 'end'])]
        log['caseid'] = log['caseid'].astype(str)
        log['caseid'] = 'Case' + log['caseid']
        sim_log = sim_log[~sim_log.task.isin(['Start', 'End', 'start', 'end'])]
        evaluator = ev.SimilarityEvaluator(log, sim_log, parms)
        metrics = ['tsd', 'day_hour_emd', 'log_mae', 'dl', 'mae']
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
        return sim_values

    @staticmethod
    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('activity', None)
        exp_desc.pop('read_options', None)
        exp_desc.pop('column_names', None)
        exp_desc.pop('one_timestamp', None)
        exp_desc.pop('reorder', None)
        exp_desc.pop('index_ac', None)
        exp_desc.pop('index_rl', None)
        exp_desc.pop('dim', None)
        exp_desc.pop('max_dur', None)
        exp_desc.pop('variants', None)
        exp_desc.pop('is_single_exec', None)
        return exp_desc


    @staticmethod
    def seconds_to_days(seconds):
        delta = timedelta(seconds=seconds)
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return days, hours, minutes, seconds

    @staticmethod
    def save_results(measurements, feature, parms):
        if measurements:
            if parms['is_single_exec']:
                output_route = os.path.join('output_files',
                                            parms['folder'],
                                            'results')
                model_name, _ = os.path.splitext(parms['model_file'])
                sup.create_csv_file_header(
                    measurements,
                    os.path.join(
                        output_route,
                        model_name + '_' + feature + '_' + parms['activity'] + '.csv'))
            else:
                if os.path.exists(os.path.join(
                        'output_files', feature + '_' + parms['activity'] + '.csv')):
                    sup.create_csv_file(
                        measurements,
                        os.path.join('output_files',
                                     feature + '_' + parms['activity'] + '.csv'),
                        mode='a')
                else:
                    sup.create_csv_file_header(
                        measurements,
                        os.path.join('output_files',
                                     feature + '_' + parms['activity'] + '.csv'))
