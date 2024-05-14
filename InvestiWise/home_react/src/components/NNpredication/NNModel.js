import React from 'react';
import LSTMModel from './Lstmprediction/LSTMModel';
import StockData from '../StockPrediction/StockData/StockData';
import StockMovingAverage from './StockMovingAverage/StockMovingAverage';
import StockDailyReturn from './DailyReturn/StockDailyReturn';

import './NNModel.css';

function NNPrediction() {
    return (
        <div style={{ overflowY: 'scroll', height: '100vh' }}>
            <LSTMModel/>
            <StockData/>
            <StockMovingAverage/>
            <StockDailyReturn/>
        </div>
    );
}

export default NNPrediction;
