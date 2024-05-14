import React, { createContext, useContext, useState } from 'react';

const ModelContext = createContext(null);

export const useModelData = () => useContext(ModelContext);

export const ModelProvider = ({ children }) => {
    const [modelData, setModelData] = useState(null);

    const loadData = async (data) => {
        try {
            const response = await fetch('http://127.0.0.1:8000/api/predict_lstm/', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await response.json();
            if (response.ok) {
                const rocData = {
                    fpr: result.prediction_results.roc_curve.fpr,
                    tpr: result.prediction_results.roc_curve.tpr,
                    auc: result.prediction_results.roc_auc
                };
                setModelData({ 
                    ...result,
                    rocData,
                    results: {
                        ...result.prediction_results,
                        stock_code: result.saved_data.stock_code,
                        prediction_days: result.saved_data.prediction_days
                    },
                    classificationReport: result.prediction_results.classification_report
                });
                console.log('Data received and saved successfully:', result);
            } else {
                console.error('Failed to fetch predictions:', result);
            }
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

    return (
        <ModelContext.Provider value={{ modelData, loadData }}>
            {children}
        </ModelContext.Provider>
    );
};
