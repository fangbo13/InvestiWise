import React, { useState } from 'react';

function StockPredictionForm() {
    const [stockCode, setStockCode] = useState('');
    const [trainingYear, setTrainingYear] = useState('');
    const [validationYears, setValidationYears] = useState('');
    const [predictionDays, setPredictionDays] = useState('');
    const [mlModel, setMlModel] = useState('LR');

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = { stockCode, trainingYear, validationYears, predictionDays, mlModel };
        
        // 添加发送数据到后端的逻辑，例如使用 fetch 或 Axios
        console.log(formData);
        // 请在此处添加 POST 请求逻辑
    };

    return (
        <div className="form-container">
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Stock Code:</label>
                    <input
                        type="text"
                        value={stockCode}
                        onChange={(e) => setStockCode(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Training Year:</label>
                    <input
                        type="number"
                        value={trainingYear}
                        onChange={(e) => setTrainingYear(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Validation Years:</label>
                    <input
                        type="number"
                        value={validationYears}
                        onChange={(e) => setValidationYears(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Prediction Days:</label>
                    <input
                        type="number"
                        value={predictionDays}
                        onChange={(e) => setPredictionDays(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Machine Learning Model:</label>
                    <select value={mlModel} onChange={(e) => setMlModel(e.target.value)}>
                        <option value="LR">Linear Regression</option>
                        <option value="RF">Random Forest</option>
                        <option value="SVM">Support Vector Machine</option>
                    </select>
                </div>
                <button type="submit">Predict</button>
            </form>
        </div>
    );
}

export default StockPredictionForm;
