import React from 'react';
import LSTMModel from './Lstmprediction/LSTMModel';
import StockData from '../StockPrediction/StockData/StockData';
import StockMovingAverage from './StockMovingAverage/StockMovingAverage'; 
import './NNModel.css';

function NNPrediction() {
    return (
        <div style={{ overflowY: 'scroll', height: '100vh' }}>
            <LSTMModel/>
            <StockData/>
            <StockMovingAverage/>
        </div>
    );
}

export default NNPrediction;
