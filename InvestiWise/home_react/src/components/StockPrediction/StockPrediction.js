import React from 'react';
import StockForm from './StockForm/StockForm';
import StockData from './StockData/StockData';
import SnowCanvas from './SnowCanvas/SnowCanvas';
import './StockPrediction.css';

function StockPrediction() {
    return (
        <div className="scrollable-container" id="scrollable-container"> 
            <StockData />
            <StockForm />
            <SnowCanvas />
        </div>
    );
}

export default StockPrediction;
