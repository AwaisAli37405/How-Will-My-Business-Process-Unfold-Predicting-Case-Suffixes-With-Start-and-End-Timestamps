
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
# from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev
from statistics import mean
from model_prediction import simulation as sim
import gc



class ModelPredictor():
    """
    This is the man class encharged of the model evaluation
    """

    def __init__(self, parms, a,s,d):
        self.output_route = os.path.join('output_files', parms['folder'])
        self.act_pred = a
        self.start_pred = s
        self.dur_pred = d

        # self.samples = dict()
        self.predictions = None
        self.sim_values = list()
        self.metrics = None

        self.parms = parms
        # self.test = self.load_log_test(self.output_route, self.parms)
        # self.test = self.test.iloc[:10]
        self.one_timestamp = self.parms['one_timestamp']

        self.get_resources()
        self._timeline_contained_by_timestamp(False)
        self.acc = self.execute_simulation()

    def get_resource_count(self, log):
        self.resource_counts = log.groupby('ac_index')['rl_index'].nunique().reset_index()
        self.resource_counts.columns = ['ac_index', 'unique_resource_count']

    def get_resources(self):
        self.log = self.load_log(self.parms)

        # self.log = self.log.iloc[:50]
        self.log = feat.add_resources(self.log, 0.85)  # same as the trianing time
        # self.log = self.log[self.log['caseid'].isin(self.log['caseid'].unique()[:5])]

        # indexes creation
        self.indexing()
        self.get_resource_count(self.log)

        self.log = self.log.sort_values(by='start_timestamp')
        self.log = feat.add_calculated_times(self, self.log)
        # now calculating the inter case features
        self.log = feat.get_resource_availability(self.log, self.parms, self.resource_counts)
        self.log = feat.get_inter_arrival_time_case(self.log)
        cols = ['caseid',  'end_timestamp', 'start_timestamp','dur','wait','ac_index', 'rl_index', 'active_cases_len', 'available_resources','inter_arrival_time']
        self.log = self.log[cols]



    def _timeline_contained_by_timestamp(self, one_timestamp: bool) -> None:
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'

        # Sort the log by the relevant timestamp
        self.log = self.log.sort_values(by=key)

        # Determine the split point (80% for training, 20% for testing)
        split_index = int(len(self.log) * 0.8)

        # Split the data
        df_train = self.log.iloc[:split_index]
        df_test = self.log.iloc[split_index:]

        self.cutt_off = df_test[key].min()
        print("cutt_off:", self.cutt_off)


        # now get the unique cases in the test
        unique_cases = df_test['caseid'].unique()
        # now get the cases with these case ids from the self.log and make them the test set
        df_test = self.log[self.log['caseid'].isin(unique_cases)]


        self.log_test = (df_test.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = (df_train.sort_values(key, ascending=True)
                          .reset_index(drop=True))




        del df_train, df_test, self.log_train, self.log
        del self.ac_index,self.rl_index
        gc.collect()




    def execute_simulation(self):

        # # enrich log
        # feat_mannager = feat.FeaturesMannager(self.parms)
        # feat_mannager.register_scaler(self.parms['model_type'],
        #                               self.model_def['vectorizer'])
        # self.log, _ = feat_mannager.calculate(self.log, self.parms['additional_columns'])

        simulation = sim.Simulation(self)
        predictions = simulation.simulate()
        # export predictions
        self.predictions = pd.DataFrame(predictions)
        self.export_predictions(0)
        # evalaute_simulation = self.evalaute_simulation(predictions)
        dl_accuracy = self._similarity_evaluation(self.predictions, 'ac')
        rl_accuracy = self._similarity_evaluation(self.predictions, 'rl')
        dur_mae = self._mae_remaining_evaluation(self.predictions, 'dur')
        wait_mae = self._mae_remaining_evaluation(self.predictions, 'wait')

        self.metrics = pd.DataFrame([{'Event_log': self.parms['model_file'], 'dl_accuracy': dl_accuracy,
                                      'rl_accuracy': rl_accuracy, 'dur_mae': dur_mae,
                                      'wait_mae': wait_mae}])

        # self.metrics = pd.DataFrame([{'Event_log': self.parms['model_file'], 'dur_mae': evalaute_simulation[0],
        #                               'wait_mae': evalaute_simulation[1]}])
        self._export_results(self.output_route)

    def evalaute_simulation(self, predictions):
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

    @staticmethod
    def load_log(params):
        params['read_options']['filter_d_attrib'] = False
        log = lr.LogReader(os.path.join('input_files', params['file_name']),
                           params['read_options'])
        log_df = pd.DataFrame(log.data)
        if set(['Unnamed: 0', 'role']).issubset(set(log_df.columns)):
            log_df.drop(columns=['Unnamed: 0', 'role'], inplace=True)
        log_df = log_df[~log_df.task.isin(['Start', 'End'])]
        return log_df


    def indexing(self):
        # Activities index creation
        # self.ac_index = self.create_index(self.log, 'task')
        # self.ac_index['start'] = 0
        # self.ac_index['end'] = len(self.ac_index)
        # self.index_ac = {v: k for k, v in self.ac_index.items()}
        # # Roles index creation
        # self.rl_index = self.create_index(self.log, 'role')
        # self.rl_index['start'] = 0
        # self.rl_index['end'] = len(self.rl_index)
        # self.index_rl = {v: k for k, v in self.rl_index.items()}

        prams, act, rl = self.load_parameters(self.output_route)
        self.ac_index = act
        self.rl_index = rl
        # Add index to the event log
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log['ac_index'] = self.log.apply(ac_idx, axis=1)
        rl_idx = lambda x: self.rl_index[x['role']]
        self.log['rl_index'] = self.log.apply(rl_idx, axis=1)

    @staticmethod
    def create_index(log_df, column):
        """Creates an idx for a categorical attribute.
        parms:
            log_df: dataframe.
            column: column name.
        Returns:
            index of a categorical attribute pairs.
        """
        temp_list = log_df[[column]].values.tolist()
        subsec_set = {(x[0]) for x in temp_list}
        subsec_set = sorted(list(subsec_set))
        alias = dict()
        for i, _ in enumerate(subsec_set):
            alias[subsec_set[i]] = i + 1
        return alias


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


    def _similarity_evaluation(self, data, feature):
        data = data.copy()

        data = data[[(feature + '_expect'), (feature + '_pred')]]


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



        mean = data['similarity'].mean()

        return mean

    def _mae_remaining_evaluation(self, data, feature):
        data = data.copy()

        # data = data[[(feature + '_expect'), (feature + '_pred'),
        #              'run_num', 'implementation', 'pref_size']]
        data = data[[(feature + '_expect'), (feature + '_pred')]]
        ae = (lambda x: np.abs(np.sum(x[feature + '_expect']) -
                               np.sum(x[feature + '_pred'])))
        data['ae'] = data.apply(ae, axis=1)
        mean = data['ae'].mean()

        return mean


    # def predict_values(self, run_num):
    #     # Predict values
    #     executioner = it.PredictionTasksExecutioner()
    #     executioner.predict(self, self.parms['activity'], run_num)

    @staticmethod
    def load_log_test(output_route, parms):
        df_test = lr.LogReader(
            os.path.join(output_route, 'parameters', 'test_log.csv'),
            parms['read_options'])
        df_test = pd.DataFrame(df_test.data)
        df_test = df_test[~df_test.task.isin(['Start', 'End'])]
        return df_test

    def load_parameters(self, output_route):
        # Loading of parameters from training
        path = os.path.join(output_route,'act',
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            parms = {k: v for k, v in data.items()}
            parms.pop('rep', None)
            distinct_parms = {**self.parms, **parms}
            if 'dim' in data.keys():
                distinct_parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
            if self.parms['one_timestamp']:
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


    # def sampling(self, sampler):
    #     sampler.register_sampler(self.parms['model_type'],
    #                              self.model_def['vectorizer'])
    #     self.samples = sampler.create_samples(
    #         self.parms, self.log, self.ac_index,
    #         self.rl_index, self.model_def['additional_columns'])
    #
    #
    # def predict(self, executioner, run_num):
    #
    #     results = executioner.predict(self.parms,
    #                                   self.model,
    #                                   self.samples,
    #                                   self.imp,
    #                                   self.model_def['vectorizer'])
    #     results = pd.DataFrame(results)
    #     self.predictions = results

    def export_predictions(self, r_num):
        # output_folder = os.path.join(self.output_route, 'results')
        if not os.path.exists(self.output_route):
            os.makedirs(self.output_route)
        self.predictions.to_csv(
            os.path.join(
                self.output_route, 'gen_' + self.parms['variant'] + '_' +
                                   self.parms['model_file'].split('.')[0] + '.csv'),
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

    # def read_model_definition(self, model_type):
    #     Config = cp.ConfigParser(interpolation=None)
    #     Config.read('models_spec.ini')
    #     #File name with extension
    #     self.model_def['additional_columns'] = sup.reduce_list(
    #         Config.get(model_type,'additional_columns'), dtype='str')
    #     self.model_def['vectorizer'] = Config.get(model_type, 'vectorizer')

    def _export_results(self, output_path) -> None:
        # Save results
        # pd.DataFrame(self.sim_values).to_csv(
        #     os.path.join(self.output_route, sup.file_id(prefix='SE_')),
        #     index=False)
        # Save logs
        # log_test = self.log[~self.log.task.isin(['Start', 'End'])]
        # log_test.to_csv(
        #     os.path.join(self.output_route, 'tst_' +
        #                  self.parms['model_file'].split('.')[0] + '.csv'),
        #     index=False)
        self.metrics.to_csv(
            os.path.join(self.output_route, 'result_' + self.parms['variant'] + '_' +
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
