// ROCModule.js
import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { useModelData } from '../Lstmprediction/ModelContext';

const ROCModule = () => {
    const { modelData } = useModelData();
    const chartRef = useRef(null);

    useEffect(() => {
        if (modelData && modelData.rocData && chartRef.current) {
            const { fpr, tpr, auc } = modelData.rocData; // Make sure these match the structure of your modelData
            const chartInstance = echarts.init(chartRef.current);
            const options = {
                backgroundColor: '#ffffff',
                title: {
                    text: 'Receiver Operating Characteristic (ROC) Curve',
                    left: 'center',
                    textStyle: {
                        color: '#333',
                        fontWeight: 'bold',
                        fontSize: 18
                    }
                },
                tooltip: {
                    trigger: 'item',
                    formatter: function (params) {
                        return `False Positive Rate (FPR): ${params.value[0].toFixed(2)}<br/>True Positive Rate (TPR): ${params.value[1].toFixed(2)}`;
                    }
                },
                grid: {
                    left: '10%',
                    right: '10%',
                    bottom: '10%',
                    top: '10%',
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    name: 'False Positive Rate (FPR)',
                    nameLocation: 'middle',
                    nameGap: 30,
                    min: 0,
                    max: 1,
                    axisLine: {
                        lineStyle: {
                            color: '#333'
                        }
                    },
                    splitLine: {
                        show: true,
                        lineStyle: {
                            color: '#ddd',
                            type: 'dashed'
                        }
                    }
                },
                yAxis: {
                    type: 'value',
                    name: 'True Positive Rate (TPR)',
                    nameLocation: 'middle',
                    nameGap: 50,
                    min: 0,
                    max: 1,
                    axisLine: {
                        lineStyle: {
                            color: '#333'
                        }
                    },
                    splitLine: {
                        show: true,
                        lineStyle: {
                            color: '#ddd',
                            type: 'dashed'
                        }
                    }
                },
                series: [{
                    type: 'line',
                    smooth: true,
                    data: fpr.map((value, index) => [value, tpr[index]]),
                    symbol: 'circle',
                    symbolSize: 8,
                    itemStyle: {
                        color: '#FF6347',
                        borderColor: '#FF6347',
                        borderWidth: 2,
                        shadowColor: 'rgba(0, 0, 0, 0.3)',
                        shadowBlur: 10
                    },
                    lineStyle: {
                        color: '#FF6347',
                        width: 3
                    }
                }],
                graphic: {
                    elements: [{
                        type: 'text',
                        style: {
                            text: `AUC: ${auc.toFixed(3)}`,
                            font: '14px Arial',
                            textAlign: 'center',
                            fill: '#666'
                        },
                        left: 'center',
                        bottom: 20
                    }]
                }
            };

            chartInstance.setOption(options);

            return () => chartInstance.dispose();
        }
    }, [modelData]);

    return (
        <div ref={chartRef} style={{ width: '100%', height: '600px', borderRadius: '10px', overflow: 'hidden' }}>
            {!modelData && <p>No ROC Data Available</p>}
        </div>
    );
};

export default ROCModule;
