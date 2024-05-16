import React from 'react';
import { Link } from 'react-router-dom'; // 确保已经安装 react-router-dom
import './Footer.css'; // 为Footer组件指定样式

function Footer() {
    return (
        <div className="footer-container">
            <Link to="/" className="footer-link">Back to Homepage</Link>
            <Link to="/predict" className="footer-link">Try Stock Prediction</Link>
            <Link to="/lstm" className="footer-link">Try LSTM Prediction</Link>

        </div>
    );
}

export default Footer;
