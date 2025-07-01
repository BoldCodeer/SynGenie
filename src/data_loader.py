import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_obj_or_path):
    if hasattr(file_obj_or_path, 'seek'):  # it's a file-like object
        file_obj_or_path.seek(0)  # reset pointer before reading

    df = pd.read_csv(file_obj_or_path)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders
