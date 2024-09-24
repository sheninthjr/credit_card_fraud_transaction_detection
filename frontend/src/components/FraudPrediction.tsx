import { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import '../App.css';

interface ResultType {
  from_account: number;
  to_account: string;
  amount: number;
  prediction_score: number;
  is_fraud: boolean;
  message: string;
}

const FraudPrediction = () => {
  const [formData, setFormData] = useState({
    fromAccount: '',
    toAccount: '',
    amount: ''
  });
  const [result, setResult] = useState<ResultType | null>(null);
  const [error, setError] = useState('');

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      const response = await axios.post<ResultType>('http://127.0.0.1:5000/predict', {
        from_account: parseInt(formData.fromAccount),
        to_account: parseInt(formData.toAccount),
        amount: parseFloat(formData.amount)
      });
      setResult(response.data);
    } catch (err) {
      setError('Error: Unable to process the request. Please check the input values and try again.');
    }
  };

  return (
    <div className="container">
      <h1>Credit Card Fraud Detection</h1>
      <form className="form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label>From Account Number:</label>
          <input
            type="number"
            name="fromAccount"
            value={formData.fromAccount}
            onChange={handleChange}
            required
            className="form-control"
            placeholder="Enter from account number"
          />
        </div>
        <div className="form-group">
          <label>To Account Number</label>
          <input
            type="number"
            name="toAccount"
            value={formData.toAccount}
            onChange={handleChange}
            className="form-control"
            placeholder="Enter to account number"
          />
        </div>
        <div className="form-group">
          <label>Amount:</label>
          <input
            type="number"
            name="amount"
            value={formData.amount}
            onChange={handleChange}
            required
            className="form-control"
            placeholder="Enter amount"
          />
        </div>
        <button type="submit" className="btn-submit">Predict</button>
      </form>

      {error && <p className="error-message">{error}</p>}

      {result && (
        <div className={`result-box ${result.is_fraud ? 'fraud' : 'safe'}`}>
          <h2>Prediction Result</h2>
          <p><strong>From Account:</strong> {result.from_account}</p>
          <p><strong>To Account:</strong> {result.to_account || 'N/A'}</p>
          <p><strong>Amount:</strong> ${result.amount}</p>
          <p><strong>Prediction Score:</strong> {result.prediction_score.toFixed(4)}</p>
          <p className="result-message">{result.message}</p>
        </div>
      )}
    </div>
  );
};

export default FraudPrediction;
