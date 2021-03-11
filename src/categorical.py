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

        self.lbl_encoders = dict()
        self.output_df = self.df.copy(deep=True)

        if self.handle_na:
            for c in cat_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-999999999")

    
    def _label_encoding(self):

        for c in self.cat_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.lbl_encoders[c] = lbl
        
        return self.output_df

    def transform(self):
        if self.encoding_type == "label":
            return self._label_encoding()
        else:
            raise Exception(f"Encoding type {self.encoding_type} not supported!")
    


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../inputs/train_cat.csv")
    cols = [c for c in df.columns if c not in ("id", "target") ]
    cat_encoder = CategoricalFeatureEncoder(df, cat_features=cols, encoding_type="label", handle_na=True)

    output_df = cat_encoder.transform()
    print(output_df.head())
            

            
        