// RiskInvestments.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import styles from './RiskInvestments.module.css';

const RiskInvestments = () => {
    const [data, setData] = useState({});
    const [loading, setLoading] = useState(true);
    const [showPopup, setShowPopup] = useState(false);
    const [inputData, setInputData] = useState({
        startDate: '2020-01-01',
        endDate: new Date().toISOString().split('T')[0],
        stockCode: ''
    });
    const [result, setResult] = useState(null);

    useEffect(() => {
        axios.get('http://127.0.0.1:8000/api/risk_coefficients/')
            .then(response => {
                setData(response.data);
            })
            .catch(error => {
                console.error('There was an error fetching the risk coefficients!', error);
            })
            .finally(() => {
                setLoading(false);
            });
    }, []);

    const togglePopup = () => {
        setShowPopup(!showPopup);
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        axios.get(`http://127.0.0.1:8000/api/calculate_risk/?startDate=${inputData.startDate}&endDate=${inputData.endDate}&stockCode=${inputData.stockCode}`)
            .then(response => {
                setResult(response.data);
            })
            .catch(error => {
                console.error('There was an error calculating the risk!', error);
            });
    };

    return (
        <div className={styles.container}>
            <div className={styles.investmentSection}>
                <div className={styles.header}>
                    <h3>Investments</h3>
                    <p>Change since last login</p>
                </div>
                <div className={styles.stocks}>
                    {loading ? (
                        <p>Loading...</p>
                    ) : (
                        Object.keys(data).slice(0, 3).map((stockKey) => (
                            <div key={stockKey} className={styles.stockCard}>
                                <div className={styles.stockCode}>{stockKey}</div>
                                <div className={styles.riskCoefficient}>
                                    {data[stockKey] ? `${(data[stockKey] * 100).toFixed(2)}%` : 'Loading...'}
                                </div>
                            </div>
                        ))
                    )}
                    <div className={styles.viewAllCard} onClick={togglePopup}>
                        <div className={styles.viewAllText}>View Other</div>
                        <div className={styles.viewAllButton}>&#x27A4;</div>
                    </div>
                </div>
            </div>
            <div className={styles.tableSection}>
                <form onSubmit={handleSubmit} className={styles.form}>
                    <div className={styles.formGroup}>
                        <label htmlFor="startDate">Start Date</label>
                        <input
                            type="date"
                            id="startDate"
                            name="startDate"
                            value={inputData.startDate}
                            onChange={handleInputChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="endDate">End Date</label>
                        <input
                            type="date"
                            id="endDate"
                            name="endDate"
                            value={inputData.endDate}
                            onChange={handleInputChange}
                            required
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="stockCode">Stock Code</label>
                        <input
                            type="text"
                            id="stockCode"
                            name="stockCode"
                            value={inputData.stockCode}
                            placeholder="Enter stock code"  // 添加占位符
                            onChange={handleInputChange}
                            required
                        />
                    </div>
                    <button type="submit" className={styles.submitButton}>Calculate Risk</button>
                </form>
                {result && (
                    <div className={styles.result}>
                        <h3>Risk Calculation Result</h3>
                        <p>{result.risk_coefficient.toFixed(4)}</p>
                    </div>
                )}
            </div>
            {showPopup && (
                <div className={styles.popup}>
                    <div className={styles.popupContent}>
                        <span className={styles.closeButton} onClick={togglePopup}>&times;</span>
                        <h3>All Stocks</h3>
                        <div className={styles.stockGrid}>
                            {Object.keys(data).map((stockKey) => (
                                <div key={stockKey} className={styles.stockCard}>
                                    <div className={styles.stockCode}>{stockKey}</div>
                                    <div className={styles.riskCoefficient}>
                                        {data[stockKey] ? `${(data[stockKey] * 100).toFixed(2)}%` : 'Loading...'}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default RiskInvestments;
