import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useState } from 'react';
import ClassificationReportModule from '../PredictionDisplay/ClassificationReportModule';
import PredictionResultsModule from '../PredictionDisplay/PredictionResultsModule';
import ROCModule from '../PredictionDisplay/ROCModule';
import './StockForm.css';

function StockForm() {
    const [stockCode, setStockCode] = useState('');
    const [trainingYear, setTrainingYear] = useState('');
    const [validationYears, setValidationYears] = useState('');
    const [predictionDays, setPredictionDays] = useState('');
    const [mlModel, setMlModel] = useState('SVM'); // Default to 'Support Vector Machine'
    const [errors, setErrors] = useState({});
    const [rocData, setRocData] = useState(null);
    const [predictionResults, setPredictionResults] = useState(null);

    const showAlert = (message) => {
        alert(message);
    };

    const validateForm = () => {
        let newErrors = {};
        if (!stockCode) newErrors.stockCode = "Please enter the stock code.";
        if (!trainingYear || trainingYear < 4 || trainingYear > 19) {
            showAlert("Training year must be between 4 and 19.");
            return false;
        }
        if (!validationYears || validationYears < 20 || validationYears > 30) {
            showAlert("Validation years must be between 20 and 30.");
            return false;
        }
        if (!predictionDays || predictionDays < 1 || predictionDays > 30) {
            showAlert("Prediction days must be between 1 and 30.");
            return false;
        }
        if (Object.keys(newErrors).length > 0) {
            setErrors(newErrors);
            return false;
        }
        return true;
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!validateForm()) {
            return; // Validation failed, stop submission.
        }

        const postData = {
            stock_code: stockCode,
            training_year: parseInt(trainingYear, 10),
            validation_years: parseInt(validationYears, 10),
            prediction_days: parseInt(predictionDays, 10),
            ml_model: mlModel,
        };

        try {
            const response = await fetch('http://localhost:8000/api/StockForm/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(postData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log("Received data:", data);  // Log to check the structure

            setRocData({
                fpr: data.prediction_results.roc_curve.fpr,
                tpr: data.prediction_results.roc_curve.tpr
            });

            setPredictionResults({
                ...data.prediction_results,  // includes roc_auc, classification_report
                pred: data.prediction_results.pred,
                stock_code: data.saved_data.stock_code,
                prediction_days: data.saved_data.prediction_days,
                ml_model: data.saved_data.ml_model,
            });

            setErrors({});
        } catch (error) {
            console.error('Error:', error);
            setErrors({ submit: 'Failed to fetch prediction data.' });
        }
    };

    return (
        <div className="stock-form-container">
            <form className="stock-form" onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="stockCode">Stock Code:</label>
                    <input
                        type="text"
                        id="stockCode"
                        value={stockCode}
                        onChange={(e) => setStockCode(e.target.value)}
                        className="form-control"
                        placeholder="Enter stock code "
                        required
                    />
                    {errors.stockCode && <p className="error-message">{errors.stockCode}</p>}
                </div>
                <div className="form-group">
                    <label htmlFor="trainingYear">Training Year (4 &lt; Year &lt; 19):</label>
                    <input
                        type="number"
                        id="trainingYear"
                        value={trainingYear}
                        onChange={(e) => setTrainingYear(e.target.value)}
                        className="form-control"
                        placeholder="Enter training year "
                        required
                    />
                    {errors.trainingYear && <p className="error-message">{errors.trainingYear}</p>}
                </div>
                <div className="form-group">
                    <label htmlFor="validationYears">Validation Years (20 or 30):</label>
                    <input
                        type="number"
                        id="validationYears"
                        value={validationYears}
                        onChange={(e) => setValidationYears(e.target.value)}
                        className="form-control"
                        placeholder="Enter validation years "
                        required
                    />
                    {errors.validationYears && <p className="error-message">{errors.validationYears}</p>}
                </div>
                <div className="form-group">
                    <label htmlFor="predictionDays">Prediction Days (1 &le; Days &lt; 30):</label>
                    <input
                        type="number"
                        id="predictionDays"
                        value={predictionDays}
                        onChange={(e) => setPredictionDays(e.target.value)}
                        className="form-control"
                        placeholder="Enter prediction days "
                        required
                    />
                    {errors.predictionDays && <p className="error-message">{errors.predictionDays}</p>}
                </div>
                <div className="form-group">
                    <label htmlFor="mlModel">ML Model:</label>
                    <div className="model-selection-buttons">
                        <button type="button" className={`btn ${mlModel === 'SVM' ? 'btn-primary' : 'btn-outline-primary'}`}
                            onClick={() => setMlModel('SVM')}>SVM</button>
                        <button type="button" className={`btn ${mlModel === 'RF' ? 'btn-primary' : 'btn-outline-primary'}`}
                            onClick={() => setMlModel('RF')}>Random Forest</button>
                    </div>
                </div>
                <button type="submit" className="btn btn-primary">Submit</button>
            </form>
            <div className="roc-module">
                <ROCModule data={rocData} auc={predictionResults && predictionResults.roc_auc ? predictionResults.roc_auc : null} />
            </div>
            <div className="side-modules">
                <div className="classification-report">
                    <ClassificationReportModule data={predictionResults} />
                </div>
                <div className="prediction-results">
                    <h3>Prediction Results</h3>
                    <PredictionResultsModule results={predictionResults} />
                </div>
            </div>
        </div>
    );
}

export default StockForm;
