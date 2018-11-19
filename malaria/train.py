# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:02:31 2018

@author: Amey Chaware

Train the model
"""
#%%
import argparse
import logging
import os
import random

import tensorflow as tf

from model.input_fn import load_data_malaria
from model.input_fn import get_iter_from_raw
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:/Users/ameyc/Documents/deep-learning-microscopes/malaria/experiments/8_bn2/',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='C:/Users/ameyc/Documents/deep-learning-microscopes/malaria/data/',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default = None, #default='C:/Users/ameyc/Documents/deep-learning-microscopes/malaria/experiments/test/best_weights',
                    help="Subdirectory of model dir or file containing the weights")
#%%
if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
#%%
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    
    # Create the two iterators over the two datasets
    data = load_data_malaria(data_dir, params)
    train_inputs = get_iter_from_raw(data, params, numsuper = params.numsuper, intensity_scale = 1/params.numsuper, training = True)  
    eval_inputs = get_iter_from_raw(data, params, numsuper = params.numsuper, intensity_scale = 1/params.numsuper, training = False) 
#%%
    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = int(params.number*params.split)
    params.eval_size = int((1-params.split)*params.number)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
#%%
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
