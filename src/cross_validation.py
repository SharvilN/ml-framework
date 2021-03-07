import pandas as pd
from sklearn import model_selection

'''
- binary classification
- multiclass classification
- multilabel classification
- single column regression
- multi column regression
- holdout
'''

class CrossValidation:

    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type='binary',
            n_folds=5,
            multilabel_delimiter=',',
            random_state=31
        ):
        self.df = df
        self.target_cols = target_cols
        self.n_targets = len(self.target_cols)
        self.problem_type = problem_type
        self.shuffle = shuffle
        self.n_folds = n_folds
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state

        if self.shuffle is True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df['kfold'] = -1

    def split(self):
        if self.problem_type in ('binary', 'multiclass'):
            if self.n_targets != 1:
                raise Exception(f'Invalid number of targets {self.n_targets} for selected problem type: {self.problem_type}') 
            
            target = self.target_cols[0]
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)
        
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                print(len(train_idx), len(val_idx))
                self.df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ('single_col_regression', 'multi_col_regression'):
            if self.n_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception(f'Invalid combination of number of targets {self.n_targets} and problem type {self.problem_type}')
            if self.n_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception(f'Invalid combination of number of targets {self.n_targets} and problem type {self.problem_type}')
            
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.n_folds)

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                print(len(train_idx), len(valid_idx))
                self.df.loc[:, 'kfold'] = fold
        
        elif self.problem_type.startswith('holdout_'):
            holdout_pctg = int(self.problem_type.split('_')[1])
            n_holdout_samples = int(len(self.df) * holdout_pctg / 100)
            self.df.loc[:n_holdout_samples, 'kfold'] = 0
            self.df.loc[n_holdout_samples: , 'kfold'] = 1
            print(n_holdout_samples)

        elif self.problem_type == 'multilabel':
            if self.n_targets != 1:
                raise Exception(f'Invalid combination of number of targets {self.n_targets} and problem type {self.problem_type}')

            targets = self.df[self.target_cols[0]].apply(lambda x: len(x.split(self.multilabel_delimiter)))
            print(targets.value_counts())
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df, y=targets)):
                print(len(train_idx), len(valid_idx))
                self.df.loc[:, 'kfold'] = fold

        else:
            raise Exception(f'Invalid problem type found : {self.problem_type}')
        return self.df

if __name__ == '__main__':
    df = pd.read_csv('../inputs/train_1.csv')
    cv = CrossValidation(df, target_cols=['attribute_ids'], shuffle=True, problem_type='multilabel', multilabel_delimiter=' ')

    df_split = cv.split()
    print(df_split.head())
    print(df_split.tail())
    print(df_split.value_counts())