import os
import joblib
import pandas as pd

from sklearn import ensemble
from sklearn import preprocessing 
from sklearn import metrics

from . import dispatcher

FOLD = int(os.environ.get('FOLD'))
TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
    train = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid = df[df.kfold == FOLD]

    ytrain = train.target.values
    yvalid = valid.target.values

    xtrain = train.drop(['id', 'target', 'kfold'], axis=1)
    xvalid = valid.drop(['id', 'target', 'kfold'], axis=1)

    label_encoders = {}
    for c in xtrain.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(xtrain[c].values.tolist() + xvalid[c].values.tolist() + test_df[c].values.tolist())
        xtrain.loc[:, c] = lbl.transform(xtrain[c].values.tolist())
        xvalid.loc[:, c] = lbl.transform(xvalid[c].values.tolist())
        label_encoders[c] = lbl
    
    # data is ready to train!
    clf = dispatcher.MODELS[MODEL]
    clf.fit(xtrain, ytrain)
    print(xtrain.shape)
    print(xvalid.shape)
    preds = clf.predict_proba(xvalid)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f'models/{MODEL}_{FOLD}_label_encoders.pkl')
    joblib.dump(clf, f'models/{MODEL}_{FOLD}.pkl')
    joblib.dump(xtrain.columns, f'models/{MODEL}_{FOLD}_columns.pkl')