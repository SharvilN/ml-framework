from sklearn import ensemble

MODELS = {
    'random_forest': ensemble.RandomForestClassifier(n_estimators=30, verbose=2),
    'extra_trees': ensemble.ExtraTreesClassifier(n_estimators=30, verbose=2)
}