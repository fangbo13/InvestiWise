import React from 'react';
import './PredictionResultsModule.css'; // Ensure the path is correct

function PredictionResultsModule({ results }) {
    if (!results) {
        return <div>Loading prediction results...</div>;
    }

    const message = results.pred === 1 ? 'UP' : 'DOWN';
    const predictionColor = results.pred === 1 ? 'go_up' : 'go_down';

    return (
        <div>
            <h2>Prediction Results</h2>
            <div className="prediction-container">
                Predictions based on the {results.ml_model} model, {results.stock_code} will go
                <span className={predictionColor}> {message} </span> on
                the {results.prediction_days}th trading day.
            </div>
        </div>
    );
}

export default PredictionResultsModule;
