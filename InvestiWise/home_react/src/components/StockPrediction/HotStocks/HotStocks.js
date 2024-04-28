import React, { useEffect, useState } from 'react';
import './HotStocks.css'; // 确保您有对应的CSS文件和样式

function HotStocks() {
    const [stocks, setStocks] = useState([]);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [startIndex, setStartIndex] = useState(0);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/hot-stocks/');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                setStocks(data);
                setIsLoading(false);
            } catch (error) {
                console.error("Error fetching data: ", error);
                setError(error.message);
                setIsLoading(false);
            }
        };

        fetchData();

        const interval = setInterval(() => {
            setStartIndex(prevIndex => (prevIndex + 8) % stocks.length);
        }, 3000); // Switch to next 8 stocks every 10 seconds

        return () => clearInterval(interval); // Clean up interval
    }, [stocks.length]);

    if (isLoading) {
        return <p>Loading stocks...</p>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <div className="hot-stocks-list">
            {stocks.slice(startIndex, startIndex + 8).map((stock, index) => (
                <div key={index} className="stock-item">
                    {/* Placeholder for stock logo */}
                    <div className="stock-logo-placeholder">{stock.symbol[0]}</div>
                    <div className="stock-symbol">{stock.symbol}</div>
                    <div className="stock-price">${stock.last_close.toFixed(2)}</div>
                    <div className={`stock-change ${stock.change_percent > 0 ? 'positive' : 'negative'}`}>
                        {stock.change_percent.toFixed(2)}%
                    </div>
                </div>
            ))}
        </div>
    );
}

export default HotStocks;
