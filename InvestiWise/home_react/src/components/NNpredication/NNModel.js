import React from 'react';
import Footer from '../StockPrediction/Footer/Footer';
import StockData from '../StockPrediction/StockData/StockData';
import StockDailyReturn from './DailyReturn/StockDailyReturn';
import LSTMModel from './Lstmprediction/LSTMModel';
import './NNModel.css';
import PredictStockPrice from './PredictStockPrice/PredictStockPrice';
import RiskInvestments from './RiskInvestments/RiskInvestments';
import StockMovingAverage from './StockMovingAverage/StockMovingAverage';

function NNPrediction() {
    return (
        <div style={{ overflowY: 'scroll', height: '100vh' }}>
            <LSTMModel/>
            <StockData/>
            <StockMovingAverage/>
            <StockDailyReturn/>
            <RiskInvestments/>
            <PredictStockPrice/>    
            <Footer /> 
        </div>
    );
}

export default NNPrediction;
