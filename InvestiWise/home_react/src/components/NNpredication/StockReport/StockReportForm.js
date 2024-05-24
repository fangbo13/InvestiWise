import React, { useState } from 'react';
import styles from './StockReportForm.module.css';

const StockReportForm = () => {
  const [stockCode, setStockCode] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
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
            value={stockCode}
            onChange={(e) => setStockCode(e.target.value)}
          />
        </div>
        <button className={styles.submitButton} type="submit">Generate Report</button>
      </form>
      {pdfUrl && (
        <a className={styles.downloadLink} href={pdfUrl} download={`${stockCode}_report.pdf`}>Download PDF Report</a>
      )}
    </div>
  );
};

export default StockReportForm;
