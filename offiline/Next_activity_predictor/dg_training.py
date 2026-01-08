# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_training import model_trainer as tr


# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file_name', '-m': 'model_family',
              '-e': 'max_eval', '-o': 'opt_method'}
    return switch.get(opt)


def main(argv):
    parameters = dict()
    column_names = {'Case_ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user',
                    'Start_Time': 'start_timestamp',
                    'End_Time': 'end_timestamp'}
    parameters['one_timestamp'] = False  # Change this if only one timestamp in the log Else False
    parameters['read_options'] = {
        'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp']}
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        parameters['file_name'] = 'Production.csv'
        parameters['model_family'] = 'lstm'
        parameters['opt_method'] = 'bayesian'  # 'rand_hpc', 'bayesian'
        parameters['max_eval'] = 1
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:m:e:o:",
                                    ['file_name=', 'model_family=',
                                     'max_eval=', 'opt_method='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['max_eval']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 64  # Usually 32/64/128/256
    parameters['norm_method'] = ['max', 'lognorm']
    parameters['imp'] = 2
    parameters['epochs'] = 2
    parameters['n_size'] = [5, 10, 15,20,25]
    parameters['l_size'] = [50,100,150]
    parameters['lstm_act'] = ['selu', 'tanh']
    if parameters['model_family'] == 'lstm':
        parameters['model_type'] = ['concatenated'] # to run other architectures change it to any of the following ['specialized','concatenated','shared_cat']

    # elif parameters['model_family'] == 'lstm_cx': # for intracase features need to run_cx
    #     parameters['model_type'] = ['concatenated_cx'] #['shared_cat_cx', 'concatenated_cx']

    parameters['dense_act'] = ['linear']
    parameters['optim'] = ['Nadam','SGD','Adagrad']


    if parameters['model_type'] == 'simple_gan':
        parameters['gan_pretrain'] = False
    parameters.pop('model_family', None)
    # Train models
    tr.ModelTrainer(parameters)


if __name__ == "__main__":
    main(sys.argv[1:])
