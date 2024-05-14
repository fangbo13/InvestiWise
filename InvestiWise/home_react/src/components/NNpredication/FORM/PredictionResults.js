import React from 'react';
import { useModelData } from '../Lstmprediction/ModelContext';
import './PredictionResults.css';

function PredictionResultsModule() {
    const { modelData } = useModelData();
    if (!modelData || !modelData.results) {
        return <div>Loading prediction results...</div>;
    }

    const { predictions, stock_code, prediction_days } = modelData.results;

    const lastPrediction = predictions[predictions.length - 1];
    const message = lastPrediction === 1 ? 'UP' : 'DOWN';
    const predictionColor = lastPrediction === 1 ? 'go_up' : 'go_down';

    return (
        <div>
            <h2>Prediction Results</h2>
            <div className="prediction-container">
                Predictions based on the LSTM model, {stock_code} will go
                <span className={`prediction-result ${predictionColor}`}> {message} </span> on
                the {prediction_days}th trading day.
            </div>
        </div>
    );
}

export default PredictionResultsModule;
