import os
import joblib
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import preprocessing 
from sklearn import metrics

from . import dispatcher

N_FOLDS = int(os.environ.get('N_FOLDS'))
TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')

def predict():
    predictions = None

    for FOLD in range(N_FOLDS):
        test = pd.read_csv(TEST_DATA)
        cols = joblib.load(os.path.join(f'models/{MODEL}_{FOLD}_columns.pkl'))
        label_encoders = joblib.load(os.path.join(f'models/{MODEL}_{FOLD}_label_encoders.pkl'))
        for c in cols:
            lbl = label_encoders[c]
            test.loc[:, c] = lbl.transform(test[c].values.tolist())
            
        clf =  joblib.load(os.path.join(f'models/{MODEL}_{FOLD}.pkl'))
        preds = clf.predict_proba(test[cols])[:, 1]
        print(preds)
        print(preds.shape)

        if FOLD == 0: predictions = preds
        else: predictions += preds

    print(predictions)
    print(predictions.shape)
    predictions /= N_FOLDS

    subm = pd.DataFrame(np.column_stack((test.id.values, predictions)), columns=['id', 'target'])
    subm['id'] = subm['id'].astype(int)
    return subm

if __name__ == '__main__':
    submission = predict()
    submission.to_csv(f'models/{MODEL}.csv', index=False)
