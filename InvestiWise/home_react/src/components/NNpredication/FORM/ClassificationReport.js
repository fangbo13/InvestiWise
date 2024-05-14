import React from 'react';
import { useModelData } from '../Lstmprediction/ModelContext';  // Adjust the path as necessary
import './ClassificationReport.css';

function ClassificationReportModule() {
    const { modelData } = useModelData();  // Retrieve the shared data from context
    const data = modelData ? modelData.classificationReport : null;  // Ensure you adjust according to your data structure

    const renderTableData = () => {
        if (!data) {
            return <tr><td colSpan="6" className="no-data">No data available</td></tr>;
        }

        const lines = data.split('\n').filter(line => line.trim().length > 0);

        return lines.slice(1).map((line, index) => {
            const items = line.split(/\s+/).filter(Boolean);
            if (index === 0 || index === 1) {
                return (
                    <tr key={index}>
                        <td></td>
                        {items.map((item, idx) => (
                            <td key={idx}>{item}</td>
                        ))}
                    </tr>
                );
            } else if (index === 2) {
                return (
                    <tr key={index}>
                        <td></td>
                        <td>{items[0]}</td>
                        <td></td>
                        <td></td>
                        <td>{items[1]}</td>
                        {items.slice(2).map((item, idx) => (
                            <td key={idx + 5}>{item}</td>
                        ))}
                    </tr>
                );
            } else {
                return (
                    <tr key={index}>
                        {items.map((item, idx) => (
                            <td key={idx}>{item}</td>
                        ))}
                    </tr>
                );
            }
        });
    };

    const renderTableHeader = () => {
        if (!data) return null;

        const headerLine = data.split('\n')[0];
        const headers = headerLine.split(/\s+/).filter(Boolean);

        return (
            <tr>
                <th></th>
                <th></th>
                {headers.map((header, index) => (
                    <th key={index}>{header}</th>
                ))}
            </tr>
        );
    };

    return (
        <div className="classification_report">
            <h2>Classification Report</h2>
            <table className="table table-striped">
                <thead>
                    {renderTableHeader()}
                </thead>
                <tbody>
                    {renderTableData()}
                </tbody>
            </table>
        </div>
    );
}

export default ClassificationReportModule;
