import * as echarts from 'echarts';
import React, { useEffect, useRef } from 'react';

const ROCModule = ({ data, auc }) => {
    const chartRef = useRef(null);

    useEffect(() => {
        if (data && data.fpr && data.tpr && chartRef.current) {
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
                    left: '5%',
                    right: '10%',
                    bottom: '5%',
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    name: 'False Positive Rate (FPR)',
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
                    data: data.fpr.map((fpr, index) => [fpr, data.tpr[index]]),
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
                }]
            };

            chartInstance.setOption(options);

            return () => {
                chartInstance.dispose();
            };
        }
    }, [data]);

    return (
        <div 
            ref={chartRef} 
            style={{
                width: '100%', 
                height: '400px', 
                borderRadius: '10px', 
                overflow: 'hidden'
            }}>
                <h2>ROC Curve of Test Set</h2>
            {!data && <p>No ROC Data Available</p>}
            {auc && <p style={{ textAlign: 'center', marginTop: '10px' }}>ROC AUC: {auc.toFixed(3)}</p>}
        </div>
    );
};

export default ROCModule;
