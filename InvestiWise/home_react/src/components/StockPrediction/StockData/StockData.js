import React, { useState } from 'react';
import StockChart from './StockChart'; // 引入StockChart组件
import 'echarts/lib/component/dataZoom'; // 引入dataZoom组件
import HotStocks from '../HotStocks/HotStocks'; // 引入HotStocks组件
import './StockData.css';

function StockData() {
    const [stockCode, setStockCode] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [chartData, setChartData] = useState(null); // 用于存储从后端获取的图表数据

    const handleSubmit = async (event) => {
        event.preventDefault();
        const apiUrl = 'http://127.0.0.1:8000/api/stock_data/';
        const data = {
            stock_code: stockCode,
            start_date: startDate,
            end_date: endDate,
        };

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            const result = await response.json();
            if (response.ok) {
                setChartData(result.stock_data); // 确保这里使用的是后端返回数据中的正确属性
                console.log('ChartData Updated:', chartData); // 这个log可能不会立即显示最新状态，因为setState是异步的
            } else {
                throw new Error(result.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`Failed to submit data: ${error.message}`);
        }
    };

    return (
        <div className="container">
            {/* 输入区域 */}
            <div className="input-container">
                <form onSubmit={handleSubmit}>
                    <div className="mb-3">
                        <label className="form-label" htmlFor="stockCode">Stock Code:</label>
                        <input
                            type="text"
                            id="stockCode"
                            value={stockCode}
                            onChange={(e) => setStockCode(e.target.value)}
                            className="form-control"
                            required
                        />
                    </div>
                    <div className="row">
                        <div className="col-md-6 mb-3">
                            <label className="form-label" htmlFor="startDate">Start Date:</label>
                            <input
                                type="date"
                                id="startDate"
                                value={startDate}
                                onChange={(e) => setStartDate(e.target.value)}
                                className="form-control"
                                required
                            />
                        </div>
                        <div className="col-md-6 mb-3">
                            <label className="form-label" htmlFor="endDate">End Date:</label>
                            <input
                                type="date"
                                id="endDate"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="form-control"
                                required
                            />
                        </div>
                    </div>
                    <button type="submit" className="btn btn-primary">Submit</button>
                </form>
            </div>

            {/* 图表显示区域 */}
            <div className="chart-container">
                {chartData ? <StockChart data={chartData} /> : <div className="no-data">No chart data available</div>}
            </div>
            
            {/* 热门股票列表 */}    
            <div className="hot-stocks-container">
                <HotStocks />
            </div>
        </div>
    );
}

export default StockData;
