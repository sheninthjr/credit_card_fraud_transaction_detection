import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
import joblib
from flask_cors import CORS  

model = load_model('fraud_detection_model.h5')
scaler = joblib.load('scaler.pkl')

file_path = '/home/sheninthjr/Projects/credit-card-fraud/training.csv'
df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', encoding='utf-8')
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
df.dropna(subset=['trans_date_trans_time'], inplace=True)

app = Flask(__name__)
CORS(app) 

def predict_transaction(from_account, to_account, amount, df, scaler, model):
    numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    mean_values = df[numeric_features].mean()
    feature_names = numeric_features

    new_transaction_values = {col: mean_values[col] for col in feature_names if col != 'amt'}
    new_transaction_values['amt'] = amount
    new_transaction_values['cc_num'] = from_account

    transaction_df = pd.DataFrame([new_transaction_values])
    transaction_df = transaction_df.reindex(columns=feature_names, fill_value=0)

    X_new_transaction = scaler.transform(transaction_df)
    X_new_transaction = np.reshape(X_new_transaction, (X_new_transaction.shape[0], 1, X_new_transaction.shape[1]))

    prediction = model.predict(X_new_transaction)
    return prediction[0][0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No input data provided.'}), 400

    from_account = data.get('from_account')
    to_account = data.get('to_account', '')  
    amount = data.get('amount')

    if not from_account or not isinstance(from_account, (int, float)):
        return jsonify({'error': 'Invalid or missing "from_account" field.'}), 400
    if not amount or not isinstance(amount, (int, float)):
        return jsonify({'error': 'Invalid or missing "amount" field.'}), 400

    prediction = predict_transaction(from_account, to_account, amount, df, scaler, model)
    is_fraud = prediction >= 0.0025
    result = "The transaction is likely fraudulent." if is_fraud else "The transaction is not likely fraudulent."
    
    return jsonify({
        'from_account': from_account,
        'to_account': to_account,
        'amount': amount,
        'prediction_score': float(prediction),
        'is_fraud': bool(is_fraud),
        'message': result
    })

if __name__ == '__main__':
    app.run(debug=True)
