export TRAINING_DATA=inputs/train.csv
export TEST_DATA=inputs/test.csv

export N_FOLDS=5
export MODEL=$1

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train

python -m src.predict