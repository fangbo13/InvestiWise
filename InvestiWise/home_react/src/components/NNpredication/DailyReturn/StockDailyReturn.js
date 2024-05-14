import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Chart, registerables } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'chartjs-adapter-date-fns';
import styles from './StockDailyReturn.module.css';

Chart.register(...registerables, zoomPlugin);

function StockDailyReturn() {
    const [stockCode, setStockCode] = useState('AAPL');
    const [startDate, setStartDate] = useState('2020-01-01');
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [shouldFetchData, setShouldFetchData] = useState(false);
    const chartRef = useRef(null);
    const chartInstance = useRef(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://127.0.0.1:8000/api/daily-return/?stock_code=${stockCode}&start_date=${startDate}&end_date=${endDate}`);
            const data = await response.json();
            const parsedData = JSON.parse(data.data);
            setData(parsedData);
        } catch (error) {
            console.error('Error fetching the data:', error);
        } finally {
            setLoading(false);
        }
    }, [stockCode, startDate, endDate]);

    useEffect(() => {
        if (shouldFetchData) {
            fetchData();
            setShouldFetchData(false);
        }
    }, [shouldFetchData, fetchData]);

    useEffect(() => {
        if (data && !loading) {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }

            const ctx = chartRef.current.getContext('2d');
            const chartData = {
                labels: Object.keys(data['Adj Close']),
                datasets: [
                    {
                        label: 'Adj Close',
                        data: Object.values(data['Adj Close']),
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        fill: false
                    },
                    {
                        label: 'Daily Return',
                        data: Object.values(data['Daily Return']),
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1,
                        fill: false
                    }
                ]
            };

            chartInstance.current = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'month'
                            }
                        }
                    },
                    plugins: {
                        zoom: {
                            zoom: {
                                wheel: {
                                    enabled: true, // Enable zooming with the mouse wheel
                                },
                                pinch: {
                                    enabled: true, // Enable zooming with pinch gestures
                                },
                                mode: 'xy', // Enable zooming along both axes
                            },
                            pan: {
                                enabled: true, // Enable panning
                                mode: 'xy', // Enable panning along both axes
                            }
                        }
                    }
                }
            });
        }
    }, [data, loading]);

    const handleSubmit = (event) => {
        event.preventDefault();
        setShouldFetchData(true);
    };

    const handleResetZoom = () => {
        if (chartInstance.current) {
            chartInstance.current.resetZoom();
        }
    };

    return (
        <div className={styles.container}>
            <div className={styles.formContainer}>
                <h2>Stock Daily Return</h2>
                <form onSubmit={handleSubmit}>
                    <div className={styles.formGroup}>
                        <label>
                            Stock Code:
                            <input type="text" value={stockCode} onChange={e => setStockCode(e.target.value)} />
                        </label>
                    </div>
                    <div className={styles.formGroup}>
                        <label>
                            Start Date:
                            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
                        </label>
                    </div>
                    <div className={styles.formGroup}>
                        <label>
                            End Date:
                            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
                        </label>
                    </div>
                    <button type="submit" className={styles.button}>Get Daily Return</button>
                    <button type="button" className={styles.button} onClick={handleResetZoom}>Reset Zoom</button>
                </form>
            </div>
            <div className={styles.chartContainer}>
                {loading ? <div className={styles.loading}>Loading...</div> : data ? <canvas ref={chartRef} id="myChart" width="400" height="200"></canvas> : null}
            </div>
        </div>
    );
}

export default StockDailyReturn;
