import React from 'react';
import './App.css';
import HomePage from './components/HomePage/HomePage';  // 确保路径正确
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';


function App() {
  return (
    <div className="App">
      <HomePage />
    </div>
  );
}

export default App;
