from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, cat_cols, num_cols, label_encoders, scaler
