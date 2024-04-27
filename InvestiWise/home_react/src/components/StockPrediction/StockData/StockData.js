import React, { useState } from 'react';
import StockChart from './StockChart'; // 引入StockChart组件
import 'echarts/lib/component/dataZoom';  // 引入dataZoom组件
import './StockData.css';
import 'bootstrap/dist/css/bootstrap.min.css';

function StockData() {
    const [stockCode, setStockCode] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [chartData, setChartData] = useState(null); // 用于存储从后端获取的图表数据

    const handleSubmit = async (event) => {
        event.preventDefault();
        const apiUrl = 'http://127.0.0.1:8000/api/stock_data/';
        const data = {
            stock_code_input: stockCode, // 修改类名
            start_date_input: startDate, // 修改类名
            end_date_input: endDate // 修改类名
        };

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            if (response.ok) {
                setChartData(result.stock_data); // 确保这里使用的是后端返回数据中的正确属性
                console.log('ChartData Updated:', chartData);  // 这个log可能不会立即显示最新状态，因为setState是异步的
            } else {
                throw new Error(result.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`Failed to submit data: ${error.message}`);
        }
    };

    return (
        <div className="stock-data-container">
            <div className="stock-data-form-container">
                <form onSubmit={handleSubmit}>
                    <div className="form-group stock-code-input-group">
                        <label htmlFor="stockCode">Stock Code:</label>
                        <input
                            type="text"
                            id="stockCode"
                            value={stockCode}
                            onChange={(e) => setStockCode(e.target.value)}
                            required
                            className="form-control stock-code-input" 
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="startDate">Start Date:</label>
                        <input
                            type="date"
                            id="startDate"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            required
                            className="form-control"
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="endDate">End Date:</label>
                        <input
                            type="date"
                            id="endDate"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            required
                            className="form-control"
                        />
                    </div>
                    <button type="submit" className="btn btn-primary btn-block">Submit</button>
                </form>
            </div>

            {/* 图表显示区域 */}
            <div className="stock-data-chart-container">
                {chartData ? <StockChart data={chartData} /> : <div>No chart data available</div>}
            </div>
        </div>
    );
}

export default StockData;
