import * as echarts from 'echarts';
import React, { useEffect, useRef } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';


const StockChart = ({ data }) => {
    const chartRef = useRef(null);

    useEffect(() => {
        console.log(data);  // 确认传入的数据结构
        if (data && data.dates && data.prices) {
            const chartInstance = echarts.init(chartRef.current);
            const option = {
                title: {
                    text: 'Stock Closing Prices'
                },
                tooltip: {
                    trigger: 'axis'
                },
                xAxis: {
                    type: 'category',
                    data: data.dates
                },
                yAxis: {
                    type: 'value'
                },
                dataZoom: [  // 这里添加了dataZoom配置
                {
                    type: 'slider',  // 这个 dataZoom 组件是 slider 型 dataZoom 组件
                    start: 0,       // 左边在 0% 的位置。
                    end: 100        // 右边在 100% 的位置。
                },
                {
                    type: 'inside',  // 这个 dataZoom 组件是 inside 型 dataZoom 组件
                    start: 0,       // 左边在 0% 的位置。
                    end: 100        // 右边在 100% 的位置。
                }
            ],
                series: [{
                    data: data.prices,
                    type: 'line',
                    smooth: true
                }]
            };
            chartInstance.setOption(option);
            return () => {
                chartInstance.dispose();
            };
        }
    }, [data]);
    

    return <div ref={chartRef} style={{ width: '100%', height: '400px' }}></div>;
};

export default StockChart;
