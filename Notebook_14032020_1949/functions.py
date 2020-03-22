#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:22:15 2020

@author: 
"""


from skmultiflow import *

from skmultiflow.core import BaseSKMObject, ClassifierMixin

from skmultiflow.utils import check_random_state

from skmultiflow.utils import get_dimensions

import numpy as np

import pandas as pd

import random

from sklearn.metrics import confusion_matrix

import sys

import time

class Comparison:
    
  def __init__(self):
    
        super().__init__()

  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models with differents values for parameters
  def run_comparison(data, window = 100, estimators = 50, anomaly = 0.5, drift_rate = 0.3, output_file = 'results'):
      
      # = data = SEAGenerator(classification_function=0, noise_percentage=0.7, random_state=1)
    models = [HalfSpaceTrees(n_features=stream.n_features, window_size=window, 
    n_estimators=estimators, 
    size_limit=0.1*100, 
    anomaly_threshold=anomaly,
    depth=15, 
    random_state=2),
    
    IsolationForestStream(
    window_size=window, n_estimators=estimators,
     anomaly_threshold=anomaly, 
     drift_threshold=drift_rate,
      random_state=None)]
    
    
      # Setup the evaluator
      evaluator = EvaluatePrequential(pretrain_size=1, max_samples=1000, show_plot=True, 
      metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 'running_time','model_size'], 
      batch_size=1, output_file = 'results_test.csv')
      # 4. Run the evaluation
      evaluator.evaluate(stream=stream, model=models, model_names=['HSTrees','iForestASD'])
      return 