# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:08:25 2021

@author: Manuel Camargo
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_prediction import model_predictor as pr
from model_prediction import activity_predictor as a
from model_prediction import start_predictor as s
from model_prediction import dur_predictor as d


# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-a': 'activity', '-c': 'folder',
              '-b': 'model_file', '-v': 'variant', '-r': 'rep'}
    return switch.get(opt)


def main(argv):
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parameters['one_timestamp'] = False # Change this if only one timestamp in the log Else False
    parameters['is_single_exec'] = False
    parameters['read_options'] = {
        'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp'],
        'filter_d_attrib': False}
    # Parameters settled manually

    # parameters['file_name'] = 'p2p_contention.csv'
    # parameters['model_file'] = 'p2p_contention.h5'
    #
    # parameters['activity_predictor'] = 'p2p/act_predictor'
    # parameters['start_predictor'] = 'p2p/start_predictor'
    # parameters['dur_predictor'] = 'p2p/dur_predictor'
    # parameters['folder'] = 'p2p'



    opts, _ = getopt.getopt(argv, "ho:a:f:c:b:v:r:e:",
                            ['one_timestamp=', 'activity=', 'folder=',
                             'model_file=', 'variant=', 'rep=', 'file_name='])
    for opt, arg in opts:
        key = catch_parameter(opt)
        if key in ['rep']:
            parameters[key] = int(arg)
        else:
            parameters[key] = arg

    parameters['file_name'] = parameters['folder']+'.csv'
    parameters['activity_predictor'] = parameters['folder']+'/act'
    parameters['start_predictor'] = parameters['folder']+'/start'
    parameters['dur_predictor'] = parameters['folder']+'/dur'



    #now print all
    print(parameters['file_name'])
    print(parameters['activity_predictor'] )
    print(parameters['start_predictor'] )
    print(parameters['dur_predictor'] )

    # parameters['activity'] =

    # parameters['start_predictor'] = ''
    # parameters['dur_predictor'] = ''




    # print(parameters['folder'])
    # print(parameters['model_file'])

    act_holder = a.ActivityPredictor(parameters)
    start_holder = s.StartPredictor(parameters)
    dur_holder = d.DurPredictor(parameters)

    pr.ModelPredictor(parameters, act_holder, start_holder, dur_holder)


if __name__ == "__main__":
    main(sys.argv[1:])
