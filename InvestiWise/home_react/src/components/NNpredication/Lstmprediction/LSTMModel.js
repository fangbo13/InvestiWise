import React from 'react';
import './LSTMModel.css';
import Input from '../FORM/input';
import Roc from '../FORM/Roc';
import ClassificationReport from '../FORM/ClassificationReport';
import PredictionResults from '../FORM/PredictionResults';
import { ModelProvider } from './ModelContext';

function CustomModelComponent() {
    return (
        <ModelProvider>
            <div className="custom-model-container">
                <div className="custom-model-main">
                    <div className="custom-roc-model">
                        <Roc />
                    </div>
                    <div className="custom-input-model">
                        <Input />
                    </div>
                </div>
                <div className="custom-model-footer">
                    <div className="custom-classification-report">
                        <ClassificationReport />
                    </div>
                    <div className="custom-prediction-results">
                        <PredictionResults />
                    </div>
                </div>
            </div>
        </ModelProvider>
    );
}

export default CustomModelComponent;
