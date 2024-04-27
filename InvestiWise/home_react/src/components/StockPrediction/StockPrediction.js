import React from 'react';
import StockForm from './StockForm/StockForm';
import StockData from './StockData/StockData';
import './StockPrediction.css';

function StockPrediction() {
    return (
        <div className="scrollable-container">
            <StockData />
            <StockForm /> 
        </div>
    );
}

export default StockPrediction;
