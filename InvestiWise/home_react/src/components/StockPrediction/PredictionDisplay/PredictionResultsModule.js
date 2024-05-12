import React from 'react';
import './PredictionResultsModule.css'; // 确保路径正确

function PredictionResultsModule({ results }) {
    if (!results) {
        return <div>Loading prediction results...</div>;
    }

    const { predictions, best_params, classification_report, roc_auc, stock_code, prediction_days, ml_model } = results;

    // 示例：选择显示数组中的最后一个预测结果
    const lastPrediction = predictions[predictions.length - 1];
    const message = lastPrediction === 1 ? 'UP' : 'DOWN';
    const predictionColor = lastPrediction === 1 ? 'go_up' : 'go_down';

    return (
        <div>
            <h2>Prediction Results</h2>
            <div className="prediction-container">
                Predictions based on the {ml_model} model, {stock_code} will go
                <span className={`prediction-result ${predictionColor}`}> {message} </span> on
                the {prediction_days}th trading day.
            </div>
        </div>
    );
}

export default PredictionResultsModule;
