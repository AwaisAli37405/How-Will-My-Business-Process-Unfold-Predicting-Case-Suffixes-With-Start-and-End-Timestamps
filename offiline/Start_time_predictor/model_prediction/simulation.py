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

def define_columns(add_cols):
    columns = ['ac_index', 'rl_index', 'dur_norm']
    add_cols = [x + '_norm' if x != 'weekday' else x for x in add_cols]
    columns.extend(add_cols)

    columns.extend(['wait_norm'])
    return columns



class Simulation:

    def __init__(self, obj ):
        self.log = obj.log
        self.model = obj.model

        self.g_truth = {}
        self.predictions = {}
        self.columns = define_columns(obj.parms['additional_columns'])
        self.ac_index = obj.ac_index
        self.rl_index = obj.rl_index
        self.parameters = obj.parms
        self.vectorizer = obj.model_def['vectorizer']

        # self.simulate() # to start the simulation



    def simulate(self):

        # get the log and split it
        prefixes , suffixes = self.split_log()

        # for each prefix create a sample

        # get the dict of cases
        log_ = copy.deepcopy(self.log)

        prefix_dict = self.reformat_events(prefixes, self.columns, True)
        suffix_dict = self.reformat_events(suffixes, self.columns, False)
        log_dict = self.reformat_events(log_, self.columns, None)


        # loop over the dict and create samples
        for i, _ in enumerate(prefix_dict):
            self.predictions[prefix_dict[i]['caseid']] = {}


            spl_prefix = self.create_samples(prefix_dict[i])

            # get the suffix from the suffix dict having caseid similar to the prefix
            suffix = [x for x in suffix_dict if x['caseid'] == prefix_dict[i]['caseid']]

            if len(suffix) == 0:
                continue # if case ends before the cuttof
            else:
                spl_suffix = self.create_samples(copy.deepcopy(suffix[0]))


                ac, rl, dur, wait = self.predict_case(spl_prefix, spl_suffix)

                record = [x for x in log_dict if x['caseid'] == prefix_dict[i]['caseid']][0]

                self.predictions[prefix_dict[i]['caseid']]['activities'] = ac
                self.predictions[prefix_dict[i]['caseid']]['roles'] = rl
                self.predictions[prefix_dict[i]['caseid']]['durations'] = dur
                self.predictions[prefix_dict[i]['caseid']]['wait'] = wait
                # self.predictions[prefix_dict[i]['caseid']]['ground_truth'] = record['ac_index']
                # self.predictions[prefix_dict[i]['caseid']]['roles_truth'] = record['rl_index']
                # self.predictions[prefix_dict[i]['caseid']]['durations_truth'] = record['dur_norm']
                # self.predictions[prefix_dict[i]['caseid']]['wait_truth'] = record['wait_norm']
                self.predictions[prefix_dict[i]['caseid']]['ground_truth'] = suffix[0]['ac_index']
                self.predictions[prefix_dict[i]['caseid']]['roles_truth'] = suffix[0]['rl_index']
                self.predictions[prefix_dict[i]['caseid']]['durations_truth'] = suffix[0]['dur_norm']
                self.predictions[prefix_dict[i]['caseid']]['wait_truth'] = suffix[0]['wait_norm']

        # create the result record
        # remove any empty records in self.predictions



        results = self.create_result_record(self.parameters)




        return results

    def predict_case(self, prefix, suffix):

        ac_suf = []
        rl_suf = []
        dur_suf = []
        wait_suf = []


        # x_ac_ngram = (np.append(
        #     np.zeros(self.parameters['dim']['time_dim']),
        #     np.array(prefix['activities']),
        #     axis=0)[-self.parameters['dim']['time_dim']:]
        #               .reshape((1, self.parameters['dim']['time_dim'])))

        # Create an array of zeros with the same length as the time dimension
        zeros_array = np.zeros(self.parameters['dim']['time_dim'])

        # Convert the 'activities' prefix to a numpy array
        activities_array = np.array(prefix['activities'])

        # Append the activities array to the zeros array along the first axis
        appended_array = np.append(zeros_array, activities_array, axis=0)

        # Take the last 'time_dim' elements from the appended array
        trimmed_array = appended_array[-self.parameters['dim']['time_dim']:]

        # Reshape the trimmed array to have shape (1, time_dim)
        x_ac_ngram = trimmed_array.reshape((1, self.parameters['dim']['time_dim']))


        x_rl_ngram = (np.append(
            np.zeros(self.parameters['dim']['time_dim']),
            np.array(prefix['roles']),
            axis=0)[-self.parameters['dim']['time_dim']:]
                      .reshape((1, self.parameters['dim']['time_dim'])))

        # times input shape(1,5,1)
        times_attr_num = (prefix['times'][0].shape[0])
        x_t_ngram = np.array(
            [np.append(np.zeros(
                (self.parameters['dim']['time_dim'], times_attr_num)),
                prefix['times'], axis=0)
             [-self.parameters['dim']['time_dim']:]
             .reshape((self.parameters['dim']['time_dim'], times_attr_num))]
        )
        if self.vectorizer in ['basic']:
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
        elif self.vectorizer in ['inter']:
            inter_attr_num = (prefix['inter_attr'][0].shape[0])
            x_inter_ngram = np.array(
                [np.append(np.zeros((
                    self.parameters['dim']['time_dim'], inter_attr_num)),
                    prefix['inter_attr'], axis=0)
                [-self.parameters['dim']['time_dim']:]
                .reshape(
                    (self.parameters['dim']['time_dim'], inter_attr_num))]
            )
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]


        for _ in range(1, self.parameters['max_trace_size']):
            preds = self.model.predict(inputs)

            # here other sampling approaches can be implemented
            pos = np.argmax(preds[0][0])
            pos1 = np.argmax(preds[1][0])
            dur = preds[2][0][0]
            wait = preds[2][0][1]



            # here appending the ground truth
            next_ac = suffix['activities'].pop(0)
            next_rl = suffix['roles'].pop(0)
            next_times = np.array(suffix['times'].pop(0)).reshape(1,prefix['times'][0].shape[0])
            if self.vectorizer in ['inter']:
                next_inter = np.array(suffix['inter_attr'].pop(0)).reshape(1,prefix['inter_attr'][0].shape[0])

            if self.parameters['index_ac'][pos] == 'end':
                ac_suf.append(pos)
                rl_suf.append(pos1)
                dur_suf.append(0)
                wait_suf.append(0)
                break
            elif next_ac == self.ac_index['end']:
                ac_suf.append(next_ac)
                rl_suf.append(next_rl)
                dur_suf.append(0)
                wait_suf.append(0)
                break

            else:
                ac_suf.append(pos)
                rl_suf.append(pos1)
                dur_suf.append(dur)
                wait_suf.append(wait)

                x_ac_ngram = np.append(x_ac_ngram, [[next_ac]], axis=1)
                x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                x_rl_ngram = np.append(x_rl_ngram, [[next_rl]], axis=1)
                x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                x_t_ngram = np.append(x_t_ngram, [next_times], axis=1)#good
                x_t_ngram = np.delete(x_t_ngram, 0, 1)

                if self.vectorizer in ['inter']:
                    x_inter_ngram = np.append(x_inter_ngram, [next_inter], axis=1)
                    x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                else:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]



        return ac_suf, rl_suf, dur_suf, wait_suf


    def split_log(self):
        key = 'start_timestamp'

        # Sort the log by the relevant timestamp
        self.log = self.log.sort_values(by=key)

        # Determine the split point (80% for suffix, 20% for prefix)
        split_index = int(len(self.log) * 0.3)

        # Split the data
        prefixes = self.log.iloc[:split_index]
        suffixes = self.log.iloc[split_index:]

        cutt_off = prefixes[key].max()
        print("cutt_off:", cutt_off)

        return prefixes, suffixes

    def create_samples(self, log):

        times =  ['dur_norm', 'wait_norm']
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
        y_inter_dict = dict()

        # for i, _ in enumerate(log):
        for x in self.columns:
            # serie = [log[i][x]]
            serie = log[x]

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

        if self.vectorizer in ['inter']:
            vec['inter_attr'] = list()
            x_inter_dict = pd.DataFrame(x_inter_dict)
            for row in x_inter_dict.values:
                new_row = np.array(row)
                # new_row = np.dstack(new_row)
                # new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
                vec['inter_attr'].append(new_row)


        return vec


    def create_result_record(self, parms):
        results = list()


        for caseid in self.predictions.keys():
            if self.predictions[caseid] == {}:
                continue

            record = dict()
            # record['ac_prefix'] = self.predictions[caseid]['activities']
            record['ac_expect'] = self.predictions[caseid]['ground_truth']
            record['ac_pred'] = self.predictions[caseid]['activities']
            # record['rl_prefix'] = self.predictions[caseid]['prefixes']['roles']
            record['rl_expect'] = self.predictions[caseid]['roles_truth']
            record['rl_pred'] = self.predictions[caseid]['roles']
            #record['caseid'] = spl['prefixes']['caseid'][index] #need to be assigned of the prefix because prefdiction has been made using a prefix not a suffix


            if self.parameters['one_timestamp']:
                # record['tm_prefix'] = [self.rescale(
                #     x[0], parms, parms['scale_args'])
                #     for x in spl['prefixes']['times'][index]]
                record['tm_expect'] = [self.rescale(
                    x[0], self.parameters, self.parameters['scale_args'])
                    for x in self.predictions[caseid]['durations_truth']]
                record['tm_pred'] = [self.rescale(
                    x[0], self.parameters, self.parameters['scale_args'])
                    for x in self.predictions[caseid]['durations']]
            else:
                # # Duration
                # record['dur_prefix'] = [self.rescale(
                #     x[0], parms, parms['scale_args']['dur'])
                #     for x in spl['prefixes']['times'][index]]
                record['dur_expect'] = [self.rescale(
                    x, self.parameters, self.parameters['scale_args']['dur'])
                    for x in self.predictions[caseid]['durations_truth']]
                record['dur_pred'] = [self.rescale(
                    x, self.parameters, self.parameters['scale_args']['dur'])
                    for x in self.predictions[caseid]['durations']]
                # Waiting
                # record['wait_prefix'] = [self.rescale(
                #     x[1], parms, parms['scale_args']['wait'])
                #     for x in spl['prefixes']['times'][index]]
                record['wait_expect'] = [self.rescale(
                    x, self.parameters, self.parameters['scale_args']['wait'])
                    for x in self.predictions[caseid]['wait_truth']]
                record['wait_pred'] = [self.rescale(
                    x, self.parameters, self.parameters['scale_args']['wait'])
                    for x in self.predictions[caseid]['wait']]
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
                if x == 'ac_index':
                    if prefix:
                        serie.insert(0, self.ac_index[('start')])
                    if prefix == None:
                        serie.insert(0, self.ac_index[('start')])
                        serie.append(self.ac_index[('end')])
                    else:
                        serie.append(self.ac_index[('end')])
                elif x == 'rl_index':
                    if prefix:
                        serie.insert(0, self.rl_index[('start')])
                    if prefix == None:
                        serie.insert(0, self.rl_index[('start')])
                        serie.append(self.rl_index[('end')])
                    else:
                        serie.append(self.rl_index[('end')])
                else:
                    if prefix:
                        serie.insert(0, 0)
                    if prefix == None:
                        serie.insert(0, 0)
                        serie.append(0)
                    else:
                        serie.append(0)
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




