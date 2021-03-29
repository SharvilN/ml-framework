from sklearn import preprocessing

"""
- label encoding
- one hot encoding
- binarization

"""

class CategoricalFeatureEncoder():

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
        self.output_df = self.df.copy(deep=True)

        if self.handle_na:
            for c in self.cat_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-999999999")

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
        
    def fit_transform(self):
        if self.encoding_type == "label":
            return self._label_encoding()
        elif self.encoding_type == "binary":
            return self._binary_encoding()
        elif self.encoding_type == "ohe":
            return self._one_hot_encoding()
        else:
            raise Exception(f"Encoding type {self.encoding_type} not supported!")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_features:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-999999999")
        
        if self.encoding_type == 'label':
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return datafame
        
        elif self.encoding_type == 'binary':
            for c, lbl in self.binary_encoders.items():
                bin_vals = lbl.transform(dataframe[c].values)
                dataframe.drop(c, axis=1)
                for j in range(bin_vals.shape[1]):
                    new_col = c + f"__bin__{j}"
                    dataframe[new_col] = bin_vals[:, j]
            return dataframe
    

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