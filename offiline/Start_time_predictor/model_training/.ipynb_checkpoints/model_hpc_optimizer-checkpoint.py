# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:05:36 2021

@author: Manuel Camargo
"""
import os
import copy
import random
import itertools
import traceback
import ast

import pandas as pd
#import utils.support as sup
#import utils.slurm_multiprocess as slmp

import utils.slurm_multiprocess as slmp
import utils.support as sup

class ModelHPCOptimizer():
    """
    Hyperparameter-optimizer class
    """
       
    def __init__(self, parms, log, ac_index, rl_index):
        """constructor"""
        self.space = self.define_search_space(parms)
        self.log = copy.deepcopy(log)
        self.ac_index = ac_index
        self.rl_index = rl_index
        self.output_folder = os.path.join('..', 'output_files')
        # Load settings
        self.parms = parms
        self.temp_output = parms['output']
        
        
        
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
            os.makedirs(os.path.join(self.temp_output, 'opt_parms'))
        self.file_name = sup.file_id(prefix='OP_')
        # Results file
        if not os.path.exists(os.path.join(self.temp_output, self.file_name)):
            open(os.path.join(self.temp_output, self.file_name), 'w').close()
        
        self.conn = {'partition': 'amd',
                    'mem': str(32000),
                    'cpus': str(10),
                    'env': 'lstm-caise23',
                    'script': os.path.join('model_training', 
                                            'slurm_trainer.py')}
        self.slurm_workers = 50
        self.best_output = None
        self.best_params = dict()
        self.best_loss = 1
        
    @staticmethod
    def define_search_space(parms):
        space = list()
        listOLists = [parms['lstm_act'], 
                      parms['dense_act'], 
                      parms['norm_method'], 
                      parms['n_size'],
                      parms['l_size'], 
                      parms['optim'], 
                      parms['model_type']]
        # selection method definition
        preconfigs = list()
        for lists in itertools.product(*listOLists):
            preconfigs.append(dict(lstm_act=lists[0],
                                   dense_act=lists[1],
                                   norm_method=lists[2],
                                   n_size=lists[3],
                                   l_size=lists[4],
                                   optim=lists[5],
                                   model_type=lists[6]))
        def_parms = {
            'imp': parms['imp'], 'file': parms['file_name'],
            'batch_size': parms['batch_size'], 'epochs': parms['epochs'],
            'one_timestamp': parms['one_timestamp']}
        for config in random.sample(preconfigs, parms['max_eval']):
            space.append({**config, **def_parms})
        return space

    def export_params(self):
        configs_files = list()
        for config in self.space:
            config['ac_index'] = self.ac_index
            config['rl_index'] = self.rl_index
            conf_file = sup.file_id(prefix='CNF_', extension='.json')
            sup.create_json(
                config, os.path.join(self.temp_output, 'opt_parms', conf_file))
            configs_files.append(conf_file)
        self.log.to_csv(
            os.path.join(self.temp_output, 'opt_parms', 'train.csv'),
            index=False, encoding='utf-8')
        return configs_files

    def execute_trials(self):
        #import pdb; pdb.set_trace()
        
        configs_files = self.export_params()
        args = [{'p': config, 
                 'f': self.temp_output,
                 'r': self.file_name} for config in configs_files]
        #output_folder=self.output_folder
        mprocessor = slmp.HPC_Multiprocess(self.conn,
                                            args,
                                            self.temp_output,
                                            None,
                                        workers_num=self.slurm_workers,
                                            timeout=5)
        mprocessor.parallelize()
        try:
            self.file_name = os.path.join(self.temp_output, self.file_name)
            
            results = (pd.read_csv(self.file_name)
                       .sort_values('loss', ascending=True))
            result = results.head(1).iloc[0]
            self.best_output = result.output
            self.best_loss = result.loss
            self.best_params = results.head(1).to_dict('records')[0]
            self.best_params['scale_args'] = ast.literal_eval(
                self.best_params.get('scale_args'))
 
            opt_res = pd.read_csv(self.file_name)
            opt_res = opt_res[opt_res.output == result.output].iloc[0]
            #self.best_params['scale_args'] = ast.literal_eval(opt_res.scale_args)
            #self.best_params['dim'] = self.parms['dim']
         
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass

