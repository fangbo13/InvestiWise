import React, { useState } from 'react';
import { useModelData } from '../Lstmprediction/ModelContext';

function Input() {
    const [stockCode, setStockCode] = useState('');
    const [trainingYear, setTrainingYear] = useState('');
    const [predictionDays, setPredictionDays] = useState('');
    const { loadData } = useModelData();

    const handleSubmit = async (event) => {
        event.preventDefault();
        const data = {
            stock_code: stockCode,
            training_year: parseInt(trainingYear),
            prediction_days: parseInt(predictionDays)
        };
        const errorMessage = await loadData(data);  // Calling loadData from Context
        if (errorMessage) {
            alert(errorMessage);  // Display alert with error message
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <label htmlFor="stockCode">Stock Code:</label>
            <input id="stockCode" type="text" value={stockCode} onChange={e => setStockCode(e.target.value)} placeholder="Enter stock code" />

            <label htmlFor="trainingYear">Training Year:</label>
            <input id="trainingYear" type="number" value={trainingYear} onChange={e => setTrainingYear(e.target.value)} placeholder="Enter training year" />

            <label htmlFor="predictionDays">Prediction Days:</label>
            <input id="predictionDays" type="number" value={predictionDays} onChange={e => setPredictionDays(e.target.value)} placeholder="Enter prediction days" />

            <button type="submit">Predict</button>
        </form>
    );
}

export default Input;
