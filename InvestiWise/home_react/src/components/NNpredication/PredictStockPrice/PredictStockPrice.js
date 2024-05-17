import axios from 'axios';
import { Chart, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';
import zoomPlugin from 'chartjs-plugin-zoom';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import styles from './PredictStockPrice.module.css';

Chart.register(...registerables, zoomPlugin);

const PredictStockPrice = () => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);
    const [inputData, setInputData] = useState({
        stockCode: 'AAPL',
        daysToPredict: 7
    });
    const [chartData, setChartData] = useState({
        historical: [],
        dates: [],
        test: [],
        predictions: [],
        future_predictions: [],
        future_prediction_dates: []
    });
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const modalChartInstance = useRef(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setIsLoading(true);
        axios.get(`http://127.0.0.1:8000/api/predict_stock_price/?stockCode=${inputData.stockCode}&days=${inputData.daysToPredict}`)
            .then(response => {
                setChartData(response.data);
                setIsLoading(false);
            })
            .catch(error => {
                console.error('There was an error predicting the stock price!', error);
                setIsLoading(false);
            });
    };

    const renderChart = useCallback((ctx) => {
        const { historical, dates, test, predictions, future_predictions, future_prediction_dates } = chartData;
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [...dates, ...future_prediction_dates],
                datasets: [
                    {
                        label: 'Historical',
                        data: historical,
                        borderColor: 'blue',
                        fill: false
                    },
                    {
                        label: 'Test',
                        data: [...new Array(dates.length - test.length).fill(null), ...test],
                        borderColor: 'green',
                        fill: false
                    },
                    {
                        label: 'Predictions',
                        data: [...new Array(dates.length - predictions.length).fill(null), ...predictions],
                        borderColor: 'red',
                        fill: false
                    },
                    {
                        label: 'Future Predictions',
                        data: [...new Array(dates.length).fill(null), ...future_predictions],
                        borderColor: 'orange',
                        fill: false
                    }
                ]
            },
            options: {
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        beginAtZero: false
                    }
                },
                plugins: {
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'xy'
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                                mode: 'xy'
                            },
                            pinch: {
                                enabled: true,
                                mode: 'xy'
                            },
                            drag: false // Disable zoom on drag
                        }
                    }
                }
            }
        });
    }, [chartData]);

    useEffect(() => {
        if (chartInstance.current) {
            chartInstance.current.destroy();
        }
        if (chartRef.current) {
            const ctx = chartRef.current.getContext('2d');
            chartInstance.current = renderChart(ctx);
        }
    }, [chartData, renderChart]);

    const openModal = () => setIsModalOpen(true);
    const closeModal = () => setIsModalOpen(false);

    useEffect(() => {
        if (isModalOpen && chartRef.current) {
            if (modalChartInstance.current) {
                modalChartInstance.current.destroy();
            }
            const modalChartCtx = document.getElementById('modalChart').getContext('2d');
            modalChartInstance.current = renderChart(modalChartCtx);
        }
    }, [isModalOpen, chartData, renderChart]);

    const resetZoom = (chart) => {
        chart.resetZoom();
    };

    return (
        <div className={styles.container}>
            <div className={styles.investmentSection}>
                <div className={styles.header}>
                    <h5>Stock Prediction</h5>
                    <p>Predict future closing prices</p>
                </div>
                <div className={styles.stocks}>
                    <form onSubmit={handleSubmit} className={styles.form}>
                        <div className={styles.formGroup}>
                            <label htmlFor="stockCode">Stock Code</label>
                            <input
                                type="text"
                                id="stockCode"
                                name="stockCode"
                                value={inputData.stockCode}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                        <div className={styles.formGroup}>
                            <label htmlFor="daysToPredict">Days to Predict</label>
                            <input
                                type="number"
                                id="daysToPredict"
                                name="daysToPredict"
                                value={inputData.daysToPredict}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                        <button type="submit" className={styles.submitButton}>Predict Prices</button>
                    </form>
                </div>
            </div>
            <div className={styles.chartSection}>
                <button onClick={openModal} className={styles.enlargeButton}>Enlarge Chart</button>
                <button onClick={() => resetZoom(chartInstance.current)} className={styles.resetButton}>Reset Zoom</button>
                {isLoading ? <div className={styles.loader}></div> : <canvas ref={chartRef} />}
            </div>
            {isModalOpen && (
                <div className={styles.modalOverlay}>
                    <div className={styles.modal}>
                        <div className={styles.modalContent}>
                            <span className={styles.closeButton} onClick={closeModal}>&times;</span>
                            <button onClick={() => resetZoom(modalChartInstance.current)} className={styles.resetButton}>Reset Zoom</button>
                            <canvas id="modalChart" />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PredictStockPrice;
