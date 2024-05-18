import axios from 'axios';
import React, { useState } from 'react';
import { Cell, Legend, Pie, PieChart, Tooltip } from 'recharts';
import styles from './NewsSentiment.module.css';

const NewsSentiment = () => {
  const [company, setCompany] = useState('');
  const [sentimentData, setSentimentData] = useState(null);
  const [newsArticles, setNewsArticles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchSentiment = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`http://localhost:8000/api/sentiment/`, {
        params: { query: company }
      });
      const { sentiments, reddit_posts } = response.data;
      setSentimentData([
        { name: 'POSITIVE', value: sentiments.POSITIVE },
        { name: 'NEGATIVE', value: sentiments.NEGATIVE },
        { name: 'NEUTRAL', value: sentiments.NEUTRAL },
      ]);
      setNewsArticles(reddit_posts);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
      setIsLoading(false);
    }
  };

  const COLORS = ['#66b3ff', '#ff9999', '#99ff99'];

  return (
    <div className={styles.container}>
      <div className={styles.leftSection}>
        <div className={styles.headerContainer}>
          <div className={styles.header}>
            <h3 className={styles.headerText}>Market Sentiment Analysis</h3>
            <p className={styles.subHeaderText}>Analyze market sentiment for a company</p>
          </div>
        </div>
        <form onSubmit={(e) => { e.preventDefault(); fetchSentiment(); }} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="company">Company/Stock Name</label>
            <input
              type="text"
              id="company"
              name="company"
              value={company}
              onChange={(e) => {
                if (e.target.value.length <= 50) {
                  setCompany(e.target.value);
                }
              }}
              required
            />
            <small>{company.length}/50</small>
          </div>
          <button type="submit" className={styles.submitButton}>Analyze Sentiment</button>
        </form>
        <div className={styles.pieChartContainer}>
          {isLoading ? <div className={styles.loader}></div> : sentimentData && (
            <PieChart width={400} height={400}>
              <Pie
                data={sentimentData}
                cx={200}
                cy={200}
                labelLine={false}
                outerRadius={150}
                fill="#8884d8"
                dataKey="value"
              >
                {sentimentData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          )}
        </div>
      </div>
      <div className={styles.rightSection}>
        <h6>Related News</h6>
        <div className={styles.newsContainer}>
          <ul className={styles.newsList}>
            {newsArticles.map((article, index) => (
              <li key={index}>
                <a href={article.url} target="_blank" rel="noopener noreferrer">
                  {article.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default NewsSentiment;
