import os
import h2o
import glob
import copy
import pandas as pd
import numpy as np
from h2o.automl import H2OAutoML


h2o.init()

def automl_training(train_set, test_set):

    train = h2o.import_file(os.path.join('AutoML_Data', 'Structures_'+train_set +'.csv'))
    test= h2o.import_file(os.path.join('AutoML_Data', 'Structures_'+test_set +'.csv'))
    x = train.columns
    y = 'SG'
    x.remove(y)

    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    fold_numbers = train.modulo_kfold_column(n_folds=2)
    fold_numbers.set_names(["fold_numbers"])
    train = train.cbind(fold_numbers)

    #aml = H2OAutoML(max_models=1, seed=1, balance_classes=False, include_algos=["DRF", "DeepLearning"])
    aml = H2OAutoML(max_models=1, seed=5, balance_classes=False, include_algos=["DeepLearning"])
    aml.train(x=x, y=y, training_frame=train, fold_column='fold_numbers')

    lb = aml.leaderboard
    lb.head(rows=lb.nrows)

    print(80*'-')
    print(lb)
    print(80*'-')

    output = aml.leader.predict(test).as_data_frame()

    #Accuracy
    preds = output['predict'].to_numpy()
    probs_df = output.drop('predict', axis=1)
    targets = test[y].as_data_frame().to_numpy().flatten()
    accuracy = np.mean(preds==targets)

    #Top3 & Top5 Accuracy
    probs = probs_df.to_numpy()
    sorted_probs_idxs = np.flip(np.argsort(probs), axis=1)
    probs_logits = probs_df.columns.values
    probs_logits = np.array([int(x[1:]) for x in probs_logits])
    top3_preds = probs_logits[sorted_probs_idxs[:,:3]]
    top5_preds = probs_logits[sorted_probs_idxs[:,:5]]
    top3_correct = 0
    for i in range(3):
        top3_correct += top3_preds[:, i]==targets
    top3_accuracy = np.mean(top3_correct)
    top5_correct = 0
    for i in range(5):
        top5_correct += top5_preds[:, i]==targets
    top5_accuracy = np.mean(top5_correct)

    print(80*'-')
    print('Accuracy:', accuracy)
    print('Top 3 Accuracy:', top3_accuracy)
    print('Top 5 Accuracy:', top5_accuracy)
    print(80*'-')

automl_training('Modulo_Merged_2Folds', 'Test')
