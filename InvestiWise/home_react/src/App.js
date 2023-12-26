import React from 'react';
import './App.css';
import HomePage from './components/HomePage/HomePage';  // 确保路径正确
import Navbar from './components/Navbar/Navbar';


function App() {
  return (
    <div className="App">
      <HomePage />
      <Navbar />
    </div>
  );
}

export default App;
