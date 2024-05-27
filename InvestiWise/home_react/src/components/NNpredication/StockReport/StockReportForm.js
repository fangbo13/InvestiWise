import React, { useState } from 'react';
import styles from './StockReportForm.module.css';

const StockReportForm = () => {
  const [stockCode, setStockCode] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [loading, setLoading] = useState(false); // State to manage loading

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true); // Set loading to true when the request starts
    const response = await fetch('http://127.0.0.1:8000/api/generate_stock_report/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        stock_code: stockCode,
      }),
    });

    if (response.ok) {
      const data = await response.blob();
      const url = window.URL.createObjectURL(new Blob([data]));
      setPdfUrl(url);
    } else {
      alert('Failed to generate report');
    }
    setLoading(false); // Set loading to false when the request completes
  };

  return (
    <div className={styles.container}>
      <div className={styles.headerContainer}>
        <h1 className={styles.headerText}>Generate Stock Report</h1>
      </div>
      <form className={styles.form} onSubmit={handleSubmit}>
        <div className={styles.formGroup}>
          <label htmlFor="stockCode">Enter Stock Code:</label>
          <input
            type="text"
            id="stockCode"
            placeholder="e.g., AAPL"
            value={stockCode}
            onChange={(e) => setStockCode(e.target.value)}
          />
        </div>
        <button className={styles.submitButton} type="submit">Generate Report</button>
      </form>
      {loading && (
        <div className={styles.loadingContainer}>
          <div className={styles.loader}></div>
          <p className={styles.loadingText}>This may take 3-4 minutes to generate the report.</p>
        </div>
      )} {/* Show loader and message when loading */}
      {pdfUrl && (
        <a className={styles.downloadLink} href={pdfUrl} download={`${stockCode}_report.pdf`}>Download PDF Report</a>
      )}
    </div>
  );
};

export default StockReportForm;
