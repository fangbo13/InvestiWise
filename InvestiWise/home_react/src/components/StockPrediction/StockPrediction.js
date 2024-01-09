import React, { useState } from 'react';

function getCsrfToken() {
    // 从 cookie 中获取 CSRF 令牌的函数
    let csrfToken = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, 'csrftoken='.length) === 'csrftoken=') {
                csrfToken = decodeURIComponent(cookie.substring('csrftoken='.length));
                break;
            }
        }
    }
    return csrfToken;
}

function StockForm() {
    const [stockCode, setStockCode] = useState('');
    const [trainingYear, setTrainingYear] = useState('');
    const [validationYears, setValidationYears] = useState('');
    const [predictionDays, setPredictionDays] = useState('');
    const [mlModel, setMlModel] = useState('LR'); // 默认为 'Linear Regression'

    const handleSubmit = async (event) => {
        event.preventDefault();
        const csrfToken = getCsrfToken(); // 获取 CSRF 令牌

        const postData = {
            stock_code: stockCode,
            training_year: parseInt(trainingYear, 10),
            validation_years: parseInt(validationYears, 10),
            prediction_days: parseInt(predictionDays, 10),
            ml_model: mlModel,
        };

        try {
            const response = await fetch('http://localhost:8000/api/StockPrediction/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken, // 在请求头中包含 CSRF 令牌
                },
                body: JSON.stringify(postData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(data);
            // 处理成功响应
        } catch (error) {
            console.error('Error:', error);
            // 处理错误
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <label>
                Stock Code:
                <input
                    type="text"
                    value={stockCode}
                    onChange={(e) => setStockCode(e.target.value)}
                />
            </label>
            <label>
                Training Year:
                <input
                    type="number"
                    value={trainingYear}
                    onChange={(e) => setTrainingYear(e.target.value)}
                />
            </label>
            <label>
                Validation Years:
                <input
                    type="number"
                    value={validationYears}
                    onChange={(e) => setValidationYears(e.target.value)}
                />
            </label>
            <label>
                Prediction Days:
                <input
                    type="number"
                    value={predictionDays}
                    onChange={(e) => setPredictionDays(e.target.value)}
                />
            </label>
            <label>
                ML Model:
                <select value={mlModel} onChange={(e) => setMlModel(e.target.value)}>
                    <option value="LR">Linear Regression</option>
                    <option value="RF">Random Forest</option>
                    <option value="SVM">Support Vector Machine</option>
                    {/* 更多模型选项 */}
                </select>
            </label>
            <button type="submit">Submit</button>
        </form>
    );
}

export default StockForm;