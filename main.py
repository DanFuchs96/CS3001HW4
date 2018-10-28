#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #4 Main; Native Bayes Application
"""

import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing as pp


def main():
    # Pre-processing config
    remove_media = False
    remove_enemy = True

    # Data-set loading
    testing_data = np.array([['Temple',             'Home', 'Out', 'NBC'],
                             ['Georgia',            'Home', 'In',  'NBC'],
                             ['BostonCollege',      'Away', 'Out', 'ESPN'],
                             ['MichiganState',      'Away', 'Out', 'FOX'],
                             ['MiamiOhio',          'Home', 'Out', 'NBC'],
                             ['NorthCarolina',      'Away', 'Out', 'ABC'],
                             ['USC',                'Home', 'In',  'NBC'],
                             ['NorthCarolinaState', 'Home', 'Out', 'NBC'],
                             ['WakeForest',         'Home', 'Out', 'NBC'],
                             ['MiamiFlorida',       'Away', 'In',  'ABC'],
                             ['Navy',               'Home', 'Out', 'NBC'],
                             ['Stanford',           'Away', 'In',  'ABC']])
    training_data = np.array([['Texas',              'Home', 'Out', 'NBC'],
                              ['Virginia',           'Away', 'Out', 'ABC'],
                              ['GeorgiaTech',        'Home', 'In',  'NBC'],
                              ['UMass',              'Home', 'Out', 'NBC'],
                              ['Clemson',            'Away', 'In',  'ABC'],
                              ['Navy',               'Home', 'Out', 'NBC'],
                              ['USC',                'Home', 'In',  'NBC'],
                              ['Temple',             'Away', 'Out', 'ABC'],
                              ['PITT',               'Away', 'Out', 'ABC'],
                              ['WakeForest',         'Home', 'Out', 'NBC'],
                              ['BostonCollege',      'Away', 'Out', 'NBC'],
                              ['Stanford',           'Away', 'In',  'FOX'],
                              ['Texas',              'Away', 'Out', 'ABC'],
                              ['Nevada',             'Home', 'Out', 'NBC'],
                              ['MichiganState',      'Home', 'Out', 'NBC'],
                              ['Duke',               'Home', 'Out', 'NBC'],
                              ['Syracuse',           'Home', 'Out', 'ESPN'],
                              ['NorthCarolinaState', 'Away', 'Out', 'ABC'],
                              ['Stanford',           'Home', 'In',  'NBC'],
                              ['MiamiFlorida',       'Home', 'Out', 'NBC'],
                              ['Navy',               'Home', 'Out', 'CBS'],
                              ['Army',               'Home', 'Out', 'NBC'],
                              ['VirginiaTech',       'Home', 'In',  'NBC'],
                              ['USC',                'Away', 'In',  'ABC']])
    testing_results = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 0.0, 1.0, 0.0])
    training_results = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

    # Pre-processing Section
    enc = pp.OrdinalEncoder().fit(np.concatenate((training_data, testing_data)))
    training_data = enc.transform(training_data)
    testing_data = enc.transform(testing_data)

    if remove_media:
        training_data = np.delete(training_data, 3, 1)
        testing_data = np.delete(testing_data, 3, 1)

    if remove_enemy:
        training_data = np.delete(training_data, 0, 1)
        testing_data = np.delete(testing_data, 0, 1)

    # Apply Naive Bayesian
    model = GaussianNB()
    model.fit(training_data, training_results)
    predictions = model.predict(testing_data)

    # Statistical Analysis
    accuracy = metrics.accuracy_score(testing_results, predictions)
    precision = metrics.precision_score(testing_results, predictions)
    recall = metrics.recall_score(testing_results, predictions)
    f1 = metrics.f1_score(testing_results, predictions)

    # Scoring
    print(' > SCORECARD')
    print('Accuracy: ', accuracy)
    print('Precision:', precision)
    print('Recall:   ', recall)
    print('F1:       ', f1)

    # Show Predictions
    print('\n > PREDICTIONS')
    for pred in predictions:
        print('Win' if pred == 1.0 else 'Lose')


if __name__ == '__main__':
    main()
