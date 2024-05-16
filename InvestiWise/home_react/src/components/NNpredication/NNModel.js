import React from 'react';
import LSTMModel from './Lstmprediction/LSTMModel';
import StockData from '../StockPrediction/StockData/StockData';
import StockMovingAverage from './StockMovingAverage/StockMovingAverage';
import StockDailyReturn from './DailyReturn/StockDailyReturn';
import Footer from '../StockPrediction/Footer/Footer'; 
import './NNModel.css';
import RiskInvestments from './RiskInvestments/RiskInvestments';

function NNPrediction() {
    return (
        <div style={{ overflowY: 'scroll', height: '100vh' }}>
            <LSTMModel/>
            <StockData/>
            <StockMovingAverage/>
            <StockDailyReturn/>
            <RiskInvestments/>
            <Footer /> 
        </div>
    );
}

export default NNPrediction;
