import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import joblib

file_path = '/home/sheninthjr/Projects/credit-card-fraud/training.csv'
df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', encoding='utf-8')
print(f'Dataset size: {df.shape}')

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
df['year'] = df['trans_date_trans_time'].dt.year
df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['hour'] = df['trans_date_trans_time'].dt.hour
df['minute'] = df['trans_date_trans_time'].dt.minute
df['second'] = df['trans_date_trans_time'].dt.second
df.dropna(subset=['trans_date_trans_time'], inplace=True)
print("Cleaned 'trans_date_trans_time' column.")

def remove_outliers(df):
    numeric_df = df.select_dtypes(include=[np.number]).drop(['is_fraud'], axis=1)
    z_scores = np.abs(stats.zscore(numeric_df))
    filtered_df = df[(z_scores < 3).all(axis=1)]
    return filtered_df

df_no_outliers = remove_outliers(df)
print(f'Removed outliers. New shape: {df_no_outliers.shape}')

numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
X_numeric = df_no_outliers[numeric_features]
y = df_no_outliers['is_fraud']

mask = y.notna()
X_numeric = X_numeric[mask]
y = y[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_rnn_model(input_shape)
print("Training the model...")

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

model.save('fraud_detection_model.h5')
print("Model training complete.")
