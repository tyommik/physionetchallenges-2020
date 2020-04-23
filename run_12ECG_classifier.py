#!/usr/bin/env python

import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow
from models.turnikecg import Turnikv7

from get_ecg_features import get_ecg_features

def run_12ECG_classifier(data, header_data, classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    # features=np.asarray(get_12ECG_features(data,header_data))
    features = get_ecg_features(data)
    feats_reshape = np.expand_dims(features, axis=0)

    score = model.predict(feats_reshape)
    label = np.argmax(score, axis=1)

    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk
    img_rows, img_cols = 5000, 12
    num_classes = 9

    weights_file ='models/Turnikv7_best_model.h5'
    # loaded_model = load_model(weights_file)
    loaded_model = Turnikv7(input_shape=(img_rows, img_cols), n_classes=num_classes)
    loaded_model.load_weights(weights_file)


    return loaded_model
