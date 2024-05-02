import React, { useState } from 'react';
import './StockForm.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import ROCModule from '../PredictionDisplay/ROCModule';
import ClassificationReportModule from '../PredictionDisplay/ClassificationReportModule';
import PredictionResultsModule from '../PredictionDisplay/PredictionResultsModule';

function StockForm() {
    const [stockCode, setStockCode] = useState('');
    const [trainingYear, setTrainingYear] = useState('');
    const [validationYears, setValidationYears] = useState('');
    const [predictionDays, setPredictionDays] = useState('');
    const [mlModel, setMlModel] = useState('LR'); // 默认为 'Linear Regression'
    const [errors, setErrors] = useState({}); // 错误信息状态

    const validateForm = () => {
        let newErrors = {};
        // Add more field-based validation logic
        if (!stockCode) newErrors.stockCode = "Please enter the stock code";
        if (!trainingYear || trainingYear < 4 || trainingYear > 19) newErrors.trainingYear = "Training year must be between 4 and 19";
        if (!validationYears || validationYears < 1 || validationYears > 5) newErrors.validationYears = "Validation years must be between 1 and 5";
        if (!predictionDays || predictionDays < 1 || predictionDays >= 30) newErrors.predictionDays = "Prediction days must be between 1 and 30";
        if (Object.keys(newErrors).length > 0) {
            let errorMessage = '';
            Object.values(newErrors).forEach(error => {
                errorMessage += error + "\n";
            });
            alert(errorMessage);
            return false;
        }
        return true;
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!validateForm()) {
            return; // 验证失败，中止提交
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
            console.log(data);
            // 清除错误信息
            setErrors({});
            // 处理成功响应
        } catch (error) {
            console.error('Error:', error);
            // 处理错误
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
                    />
                    {errors.trainingYear && <p className="error-message">{errors.trainingYear}</p>}
                </div>
                <div className="form-group">
                    <label htmlFor="validationYears">Validation Years (1 &le; Years &le; 5):</label>
                    <input
                        type="number"
                        id="validationYears"
                        value={validationYears}
                        onChange={(e) => setValidationYears(e.target.value)}
                        className="form-control"
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
                    />
                    {errors.predictionDays && <p className="error-message">{errors.predictionDays}</p>}
                </div>
                <div className="form-group">
                    <label htmlFor="mlModel">ML Model:</label>
                    <select
                        id="mlModel"
                        value={mlModel}
                        onChange={(e) => setMlModel(e.target.value)}
                        className="form-control"
                    >
                        <option value="LR">Linear Regression</option>
                        <option value="RF">Random Forest</option>
                        <option value="SVM">Support Vector Machine</option>
                    </select>
                </div>
                <button type="submit" className="btn btn-primary">Submit</button>
            </form>
            <div className="roc-module">
                <ROCModule />
            </div>
            <div className="side-modules">
                <div className="classification-report">
                    <ClassificationReportModule />
                </div>
                <div className="prediction-results">
                    <PredictionResultsModule />
                </div>
            </div>
        </div>
    );
}

export default StockForm;
