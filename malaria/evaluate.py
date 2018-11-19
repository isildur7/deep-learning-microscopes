# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:45:02 2018

@author: Amey Chaware

Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import load_data_malaria
from model.input_fn import get_iter_from_raw
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:/Users/ameyc/Documents/deep-learning-microscopes/malaria/experiments/8_bn/',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='C:/Users/ameyc/Documents/deep-learning-microscopes/malaria/data/',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='C:/Users/ameyc/Documents/deep-learning-microscopes/malaria/experiments/8_bn2/best_weights/',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir

    # create the iterator over the dataset
    data = load_data_malaria(data_dir, params)
    test_inputs = get_iter_from_raw(data, params, numsuper = params.numsuper, intensity_scale = 1/params.numsuper, training = False)
    
    # specify the size of the evaluation set
    params.eval_size = len(data[5])+len(data[7])

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
