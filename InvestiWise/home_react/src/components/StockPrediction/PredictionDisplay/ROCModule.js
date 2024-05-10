import * as echarts from 'echarts';
import React, { useEffect, useRef } from 'react';

const ROCModule = ({ data, auc }) => {
    const chartRef = useRef(null);

    useEffect(() => {
        if (data && data.fpr && data.tpr && auc && chartRef.current) {
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
                    left: '10%',  // Increase from '5%' to give more space
                    right: '10%',
                    bottom: '10%',  // Adjust bottom space to ensure enough room for AUC text
                    top: '10%',     // Provide more space at the top for the title
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    name: 'False Positive Rate (FPR)',
                    nameLocation: 'middle',  // Adjust location of the axis name
                    nameGap: 30,             // Gap between the axis name and axis line
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
                    nameLocation: 'middle',  // Adjust location of the axis name
                    nameGap: 50,             // Increase the gap for better visibility
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
                        bottom: 20  // Adjust bottom position for visibility
                    }]
                }
            };

            chartInstance.setOption(options);

            return () => {
                chartInstance.dispose();
            };
        }
    }, [data, auc]);

    return (
        <div 
            ref={chartRef} 
            style={{
                width: '100%', 
                height: '570px',  // Adjust the height if necessary to fit your display
                borderRadius: '10px', 
                overflow: 'hidden'
            }}>
            {!data && <p>No ROC Data Available</p>}
        </div>
    );
};

export default ROCModule;
