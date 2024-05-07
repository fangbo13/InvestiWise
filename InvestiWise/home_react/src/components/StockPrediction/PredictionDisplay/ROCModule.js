import * as echarts from 'echarts';
import React, { useEffect, useRef } from 'react';

const ROCModule = ({ data }) => {
    const chartRef = useRef(null);

    useEffect(() => {
        if (data && data.fpr && data.tpr && chartRef.current) {
            const chartInstance = echarts.init(chartRef.current);
            const options = {
                backgroundColor: '#ffffff', // 白色背景
                title: {
                    text: 'Receiver Operating Characteristic (ROC) Curve',
                    left: 'center',
                    textStyle: {
                        color: '#333',
                        fontWeight: 'bold',
                        fontSize: 18 // 标题字号调整
                    }
                },
                tooltip: {
                    trigger: 'item',
                    formatter: function (params) {
                        return `False Positive Rate (FPR): ${params.value[0].toFixed(2)}<br/>True Positive Rate (TPR): ${params.value[1].toFixed(2)}`;
                    }
                },
                grid: { // 调整图表的内部边距
                    left: '5%', // 左边距调整为 5%
                    right: '10%', // 右边距
                    bottom: '2%', // 下边距调整为 15%
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
                        show: true, // 显示网格线
                        lineStyle: {
                            color: '#ddd', // 网格线颜色调整
                            type: 'dashed' // 虚线
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
                        show: true, // 显示网格线
                        lineStyle: {
                            color: '#ddd', // 网格线颜色调整
                            type: 'dashed' // 虚线
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
                        shadowColor: 'rgba(0, 0, 0, 0.3)', // 添加阴影效果
                        shadowBlur: 10 // 阴影模糊程度
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
                height: '570px', 
                borderRadius: '10px', 
                overflow: 'hidden'
            }}>
            {!data && <p>No ROC Data Available</p>}
        </div>
    );
};

export default ROCModule;
