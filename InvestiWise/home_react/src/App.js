import React from 'react';
import './App.css';
import HomePage from './components/HomePage/HomePage';
import StockPrediction from './components/StockPrediction/StockPrediction';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predict" element={<StockPrediction />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
