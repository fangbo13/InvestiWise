import React, { useEffect, useState } from 'react';
import './HomePage.css';

function HomePage() {
    const [settings, setSettings] = useState({ title: '', backgroundImage: '' });
    const [stocks, setStocks] = useState([]);

    useEffect(() => {
        // 获取页面设置
        fetch('http://127.0.0.1:8000/api/home/')
            .then(response => response.json())
            .then(data => {
                setSettings({
                    title: data.heading,
                    backgroundImage: `http://127.0.0.1:8000${data.home_background}`
                });
            });

        // 获取股票数据
        fetch('http://127.0.0.1:8000/api/stocks/')
            .then(response => response.json())
            .then(data => {
                const processedStocks = Object.entries(data).map(([symbol, change]) => {
                    const changeNum = parseFloat(change.split(' ')[1]);
                    return {
                        symbol,
                        change: changeNum.toFixed(2),
                        isUp: changeNum >= 0
                    };
                });
                setStocks(processedStocks);
            });
    }, []);

    const backgroundStyle = {
        backgroundImage: `url(${settings.backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
    };

    return (
        <div className="homepage-container" style={backgroundStyle}>
            <div className="title-container">
                <h1 className="title">{settings.title}</h1>
            </div>

            <div className="stock-ticker-wrapper">
                <div className="stock-ticker">
                    {stocks.map((stock, index) => (
                        <div key={index} className={`stock-item ${stock.isUp ? 'stock-up' : 'stock-down'}`}>
                            {stock.symbol} {stock.change}%
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default HomePage;
    