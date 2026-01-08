# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:35:37 2020

@author: Manuel Camargo
"""
import numpy as np
import math

import utils.support as sup


class SuffixPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'
        self.max_trace_size = 0

    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        self.spl = spl
        self.max_trace_size = params['max_trace_size']
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        sup.print_performed_task('Predicting suffixes')
        return predictor(params, vectorizer)

    def _get_predictor(self, model_type):
        # OJO: This is an extension point just incase 
        # a different predictor being neccesary
        return self._predict_suffix_shared_cat

    def top_k_sampling(self, preds, k=10):
        # If k is larger than the number of words, use all words
        k = min(k, len(preds))

        # Select the top k indices (words) based on the probabilities
        top_k_indices = np.argsort(preds)[-k:]

        # Normalize the probabilities of these top k words
        top_k_probs = preds[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)

        # Randomly select from the top_k_indices based on their probabilities
        selected_index = np.random.choice(top_k_indices, p=top_k_probs)
        return selected_index
    
    def top_percentile_sampling(self, preds, k=10):
        # Calculate number of elements that constitute the top percentile
        num_elements = int(len(preds) * (k / 100))

        # Ensure num_elements is at least 1 and not more than the length of preds
        num_elements = max(1, min(num_elements, len(preds)))

        # Select the top percentile indices based on the probabilities
        top_indices = np.argsort(preds)[-num_elements:]

        # Normalize the probabilities of these top percentile words
        top_probs = preds[top_indices]
        top_probs /= np.sum(top_probs)

        # Randomly select from the top_indices based on their probabilities
        selected_index = np.random.choice(top_indices, p=top_probs)
        return selected_index
    
    
    
    

    def nucleus_sampling(self, preds, p=0.9):
        # Sort the probabilities in descending order
        sorted_indices = np.argsort(preds)[::-1]

        # Calculate the cumulative probabilities
        cumulative_probs = np.cumsum(preds[sorted_indices])

        # Find the index where the cumulative probability exceeds p
        cutoff_index = np.where(cumulative_probs >= p)[0][0]

        # Select words from the top until the cumulative probability cutoff
        valid_indices = sorted_indices[:cutoff_index + 1]

        # Normalize the probabilities of these words
        valid_probs = preds[valid_indices]
        valid_probs /= np.sum(valid_probs)

        # Randomly select from the valid_indices based on their probabilities
        selected_index = np.random.choice(valid_indices, p=valid_probs)
        return selected_index

    
    
    
    
    def _predict_suffix_shared_cat(self, parms, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions
        results = list()



        for i, _ in enumerate(self.spl['prefixes']['activities']):
            # Activities and roles input shape(1,5)
            x_ac_ngram = (np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][i]),
                    axis=0)[-parms['dim']['time_dim']:]
                .reshape((1, parms['dim']['time_dim'])))

            x_rl_ngram = (np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][i]),
                    axis=0)[-parms['dim']['time_dim']:]
                .reshape((1, parms['dim']['time_dim'])))
           
            times_attr_num = (self.spl['prefixes']['times'][i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parms['dim']['time_dim'], times_attr_num)),
                    self.spl['prefixes']['times'][i], axis=0)
                    [-parms['dim']['time_dim']:]
                    .reshape((parms['dim']['time_dim'], times_attr_num))]
                )
            if vectorizer in ['basic']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif vectorizer in ['inter']:
                inter_attr_num = self.spl['prefixes']['inter_attr'][i].shape[1]
                x_inter_ngram = np.array([np.append(
                        np.zeros((parms['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][i],
                        axis=0)[-parms['dim']['time_dim']:].reshape((parms['dim']['time_dim'], inter_attr_num))])
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

            pref_size = len(self.spl['prefixes']['activities'][i])
            acum_dur, acum_wait = list(), list()
            ac_suf, rl_suf = list(), list()
            time_indexes = []


            for _  in range(1, self.max_trace_size):
                # pos=None
                # pos1=None
                preds = self.model.predict(inputs)
                if self.imp == 'random_choice':
                    # Use this to get a random choice following as PDF the predictions
                    pos = np.random.choice(
                        np.arange(0,len(preds[0][0])), p=preds[0][0])
                    pos1 = np.random.choice(
                        np.arange(0, len(preds[1][0])), p=preds[1][0])

                elif self.imp == 'arg_max':
                    pos = np.argmax(preds[0][0])
                    pos1 = np.argmax(preds[1][0])


                elif self.imp == 'd_action2':
                    # Use this to get the max prediction
                    counts = {}
                    for number in x_ac_ngram[0]:
                        if number in counts:
                            counts[int(number)] += 1

                        else:
                            counts[int(number)] = 1

                    if (len(x_ac_ngram) != 0):
                        

                        # get predictions at pred[0]
                        trace_length = len(x_ac_ngram[0])
                        max_trace_length = self.max_trace_size  # Replace with the actual maximum trace length in your dataset
                        balance_factor = 1 / (1 + math.exp(-(trace_length - max_trace_length / 2)))

                        sum = 0
                        for key in counts:
                            sum += (preds[0][0][key] * counts[key])

                        for x in range(len(preds[0][0])):
                            if (x in counts):
                                if (counts[x] == 0):
                                    continue
                                else:
                                    preds[0][0][x] = ((preds[0][0][x] / counts[x]) / sum) * balance_factor # changing the prob of act
                                    # preds[1][0][x] = (preds[1][0][x] / counts[x]) / sum

                    pos = np.argmax(preds[0][0])
                    # pos1 = np.argmax(preds[1][0])

                    if(len(time_indexes)!=0):
                        counts = {}
                        for number in time_indexes:
                            if number in counts:
                                counts[int(number)] += 1

                            else:
                                counts[int(number)] = 1

                        if (len(time_indexes) != 0):

                            # get predictions at pred[0]

                            sum = 0
                            for key in counts:
                                sum += (preds[1][0][key] * counts[key])

                            for x in range(len(preds[1][0])):
                                if (x in counts):
                                    if (counts[x] == 0):
                                        continue
                                    else:
                                        preds[1][0][x] = (preds[1][0][x] / counts[x]) / sum  # changing the prob of act

                        pos1 = np.argmax(preds[1][0])
                        time_indexes.append(pos1)

                    else:
                        pos1 = np.argmax(preds[1][0])
                        time_indexes.append(pos1)
                
                elif self.imp == 'nucleus':
                    pos = self.nucleus_sampling(preds[0][0], p=0.9)
                    pos1 = self.nucleus_sampling(preds[1][0], p=0.9)

                elif self.imp == 'topk':
                    pos = self.top_percentile_sampling(preds[0][0], k=10)
                    pos1 = self.top_percentile_sampling(preds[1][0], k=10)
                    

                elif self.imp == 'd_action':
                    # Use this to get the max prediction
                    counts = {}
                    for number in x_ac_ngram[0]:
                        if number in counts:
                            counts[int(number)] += 1

                        else:
                            counts[int(number)] = 1

                    if (len(x_ac_ngram) != 0):

                        # get predictions at pred[0]

                        sum = 0
                        for key in counts:
                            sum += (preds[0][0][key] * counts[key])

                        for x in range(len(preds[0][0])):
                            if (x in counts):
                                if (counts[x] == 0):
                                    continue
                                else:
                                    preds[0][0][x] = (preds[0][0][x] / counts[x]) / sum # changing the prob of act
                                    # preds[1][0][x] = (preds[1][0][x] / counts[x]) / sum

                    pos = np.argmax(preds[0][0])
                    # pos1 = np.argmax(preds[1][0])

                    if(len(time_indexes)!=0):
                        counts = {}
                        for number in time_indexes:
                            if number in counts:
                                counts[int(number)] += 1

                            else:
                                counts[int(number)] = 1

                        if (len(time_indexes) != 0):

                            # get predictions at pred[0]

                            sum = 0
                            for key in counts:
                                sum += (preds[1][0][key] * counts[key])

                            for x in range(len(preds[1][0])):
                                if (x in counts):
                                    if (counts[x] == 0):
                                        continue
                                    else:
                                        preds[1][0][x] = (preds[1][0][x] / counts[x]) / sum  # changing the prob of act

                        pos1 = np.argmax(preds[1][0])
                        time_indexes.append(pos1)

                    else:
                        pos1 = np.argmax(preds[1][0])
                        time_indexes.append(pos1)








                    # Use this to get the max prediction


                # Activities accuracy evaluation
                x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                x_t_ngram = np.append(x_t_ngram, [preds[2]], axis=1)
                x_t_ngram = np.delete(x_t_ngram, 0, 1)
                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    x_inter_ngram = np.append(x_inter_ngram, [preds[3]], axis=1)
                    x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                # Stop if the next prediction is the end of the trace
                # otherwise until the defined max_size
                ac_suf.append(pos)
                rl_suf.append(pos1)
                acum_dur.append(preds[2][0][0])
                if not parms['one_timestamp']:
                    acum_wait.append(preds[2][0][1])
                if parms['index_ac'][pos] == 'end':
                    break
            # save results
            predictions = [ac_suf, rl_suf, acum_dur]
            if not parms['one_timestamp']:
                predictions.extend([acum_wait])
            results.append(
                self.create_result_record(i, self.spl, predictions, parms, pref_size))
        sup.print_done_task()
        
        return results

    def create_result_record(self, index, spl, preds, parms, pref_size):
        record = dict()
        record['pref_size'] = pref_size
        record['ac_prefix'] = spl['prefixes']['activities'][index]
        record['ac_expect'] = spl['next_evt']['activities'][index]
        record['ac_pred'] = preds[0]
        record['rl_prefix'] = spl['prefixes']['roles'][index]
        record['rl_expect'] = spl['next_evt']['roles'][index]
        record['rl_pred'] = preds[1]
        #record['caseid'] = spl['prefixes']['caseid'][index] #need to be assigned of the prefix because prefdiction has been made using a prefix not a suffix


        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']) 
                for x in spl['prefixes']['times'][index]]
            record['tm_expect'] = [self.rescale(
                x[0], parms, parms['scale_args']) 
                for x in spl['next_evt']['times'][index]]
            record['tm_pred'] = [self.rescale(
                x, parms, parms['scale_args']) 
                for x in preds[2]]
        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur']) 
                for x in spl['prefixes']['times'][index]]
            record['dur_expect'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur']) 
                for x in spl['next_evt']['times'][index]]
            record['dur_pred'] = [self.rescale(
                x, parms, parms['scale_args']['dur']) 
                for x in preds[2]]
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait']) 
                for x in spl['prefixes']['times'][index]]
            record['wait_expect'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait']) 
                for x in spl['next_evt']['times'][index]]
            record['wait_pred'] = [self.rescale(
                x, parms, parms['scale_args']['wait']) 
                for x in preds[3]]
        return record

    @staticmethod
    def rescale(value, parms, scale_args):
        if parms['norm_method'] == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
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
