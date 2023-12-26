import React, { useEffect, useState } from 'react';
import Navbar from '../Navbar/Navbar';
import './HomePage.css';
import 'bootstrap/dist/css/bootstrap.min.css';



function HomePage() {
    const [settings, setSettings] = useState({ title: '', brand: '', introduce: '' });
    const [stocks, setStocks] = useState([]);

    useEffect(() => {
        // 获取页面设置
        fetch('http://127.0.0.1:8000/api/home/')
            .then(response => response.json())
            .then(data => {
                setSettings({
                    title: data.heading,
                    brand: data.brand_name, // Add the brand name
                    introduce: data.introduce, // Add the introduce text
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

    return (
        <div>
            <Navbar/>

            <div className="homepage-container">
                <div className="header">
                    <div className="inner-header">
                        <div className="stock-ticker-wrapper">
                            <div className="stock-ticker">
                                {stocks.map((stock, index) => (
                                    <div key={index} className={`stock-item ${stock.isUp ? 'stock-up' : 'stock-down'}`}>
                                        {stock.symbol} {stock.change}%
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="title-brand-container">
                            <h1 className="title">{settings.title}</h1>
                            <span className="brand-name">{settings.brand}</span>
                        </div>
            
                        <div className="introduce-container">
                            <p className="introduce">{settings.introduce}</p> {/* Display the introduce text */}
                        </div>
                    </div>
                    <div>
                        <svg className="waves" xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 24 150 28" preserveAspectRatio="none" shape-rendering="auto">
                            <defs>
                                <path id="gentle-wave" d="M-160 44c30 0 58-18 88-18s 58 18 88 18 58-18 88-18 58 18 88 18 v44h-352z" />
                            </defs>
                            <g className="parallax">
                                <use href="#gentle-wave" x="48" y="0"  />
                                <use href="#gentle-wave" x="48" y="3"  />
                                <use href="#gentle-wave" x="48" y="5"  />
                                <use href="#gentle-wave" x="48" y="7"  />
                            </g>
                        </svg>
                    </div>
                </div>


            </div>
        </div>
    );
}

export default HomePage;
