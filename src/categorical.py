import torch
import torch.nn as nn
from typing import List

from sklearn import model_selection

from pandas.core.frame import DataFrame

import numpy as np
from sklearn import preprocessing

class CategoricalFeatureEncoder():
    """
    Universal categorical feature encoder supports most of the common types of categorical feature encoding
    techniques

    - label encoding
    - one hot encoding
    - binarization
    - target encoding
    - entity embedding
    """

    def __init__(self, df, cat_features, encoding_type, handle_na=False):
        """
        df - pandas dataframe
        cat_features - list of column names ["ord_1", "nom_1", ...]
        encoding_type - label encoding, binary, ohe 
        """
        self.df = df
        self.cat_features = cat_features
        self.encoding_type = encoding_type
        self.handle_na = handle_na

        self.label_encoders = dict()
        self.binary_encoders= dict()
        self.ohe = None
        self.entity_encoder = None

        self.output_df = self.df.copy(deep=True)

        if self.handle_na:
            for c in self.cat_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("NONE")

    def _label_encoding(self):
        for c in self.cat_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _binary_encoding(self):
        for c in self.cat_features:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            bin_vals = lbl.transform(self.df[c].values) # array
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(bin_vals.shape[1]):
                new_col = c + f"__bin__{j}"
                self.output_df[new_col] = bin_vals[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot_encoding(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_features].values)
        ohe.transform(self.output_df[self.cat_features].values)
        return self.output_df
        
    def _entity_embeddings(self):
        self.output_df = self._label_encoding()
        for col in self.cat_features:
            num_unique_vals = self.output_df[col].nunique
            embed_dim = int(min(np.ceil(num_unique_vals/2), 50))
            

    def fit_transform(self):
        if self.encoding_type == "label":
            return self._label_encoding()
        elif self.encoding_type == "binary":
            return self._binary_encoding()
        elif self.encoding_type == "ohe":
            return self._one_hot_encoding()
        elif self.encoding_type == "target":
            return self._target_encoding()
        elif self.encoding_type == "entity":
            return self._entity_embeddings()
        else:
            raise Exception(f"Encoding type {self.encoding_type} not supported!")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_features:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("NONE")
        
        if self.encoding_type == 'label':
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        
        elif self.encoding_type == 'binary':
            for c, lbl in self.binary_encoders.items():
                bin_vals = lbl.transform(dataframe[c].values)
                dataframe.drop(c, axis=1)
                for j in range(bin_vals.shape[1]):
                    new_col = c + f"__bin__{j}"
                    dataframe[new_col] = bin_vals[:, j]
            return dataframe
    
class EntityEmbeddingExtractor(nn.Module):

    def __init__(self, df: DataFrame, cat_features: List[str]):
        super(EntityEmbeddingExtractor, self).__init__()
        self.embeddings = nn.ModuleList()
        total_embed_dim = 0
        for col in cat_features:
            num_unique_values = df[col].nunique()
            embed_dim = int(min(np.ceil(num_unique_values/2), 50))
            self.embeddings.append(nn.Embedding(num_embeddings=num_unique_values+1,
                                     embedding_dim=embed_dim))
            total_embed_dim += embed_dim
        self.linear = nn.Linear(in_features=total_embed_dim, out_features=128)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm1d(128)

    def forward(self, x):
        outputs = [embed_layer(cat_token)
                     for (cat_token, embed_layer) in zip(x, self.embeddings)]
        outputs = torch.tensor(outputs)
        print(f"output shape before flatten: {outputs.shape}")
        outputs = outputs.view(-1)
        print(f"flattened output shape: {outputs.shape}")
        y = self.linear(outputs)
        y = self.dropout(outputs)
        y = self.batchnorm(outputs)
        return y

if __name__ == "__main__":
    import pandas as pd
    train_df = pd.read_csv("../inputs/train_cat.csv")
    test_df = pd.read_csv("../inputs/test_cat.csv")
    test_df["target"] = -1

    train_idx = train_df.id.values
    test_idx = test_df.id.values

    full_df = pd.concat([train_df, test_df])
    cols = [c for c in full_df.columns if c not in ("id", "target")]
    cat_encoder = CategoricalFeatureEncoder(full_df.head(), cat_features=cols, encoding_type="ohe", handle_na=True)
    full_df_transformed = cat_encoder.fit_transform()

    train = full_df_transformed[full_df_transformed["id"].isin(train_idx)].reset_index(drop=True)
    test = full_df_transformed[full_df_transformed["id"].isin(test_idx)].reset_index(drop=True)

    print(train.shape)
    print(test.shape)

    print(full_df.head())

    df = train_df
    X, y = df.drop(["target", "id"], axis=1), df["target"]
    xtrain, xval, ytrain, yval = \
    model_selection.train_test_split(X, y, test_size=0.2)
    model = EntityEmbeddingExtractor(xtrain, X.columns.values)

    print(model)