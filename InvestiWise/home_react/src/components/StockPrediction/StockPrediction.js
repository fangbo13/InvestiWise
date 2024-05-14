import React from 'react';
import StockForm from './StockForm/StockForm';
import StockData from './StockData/StockData';
import SnowCanvas from './SnowCanvas/SnowCanvas';
import Footer from './Footer/Footer'; 
import './StockPrediction.css';

function StockPrediction() {
    return (
        <div className="scrollable-container" id="scrollable-container"> 
            <StockData />
            <StockForm />
            <SnowCanvas />
            <Footer /> 
        </div>
    );
}

export default StockPrediction;
