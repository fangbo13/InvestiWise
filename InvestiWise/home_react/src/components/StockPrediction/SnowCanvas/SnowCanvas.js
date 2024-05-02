import React, { useEffect, useRef } from 'react';

const SnowCanvas = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // 设置初始画布大小
        const setCanvasSize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight; // 使用视口的高度来确保覆盖整个可见区域
        };

        window.addEventListener('resize', setCanvasSize);
        setCanvasSize();

        let snowflakes = [];
        const snowflakeCount = 150;

        for (let i = 0; i < snowflakeCount; i++) {
            snowflakes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 6 + 1, // 将半径调大一点
                density: Math.random() * 0.5 + 0.5,
                xMovement: Math.random() * 2 - 1
            });
        }

        const getGradient = (ctx, flake) => {
            let gradient = ctx.createRadialGradient(flake.x, flake.y, 0, flake.x, flake.y, flake.radius);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
            return gradient;
        };

        const drawSnowflakes = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 绘制雪花
            snowflakes.forEach(flake => {
                ctx.beginPath();
                ctx.fillStyle = getGradient(ctx, flake); // 添加渐变效果
                ctx.arc(flake.x, flake.y, flake.radius, 0, Math.PI * 2);
                ctx.fill();
            });
            moveSnowflakes();
        };

        const moveSnowflakes = () => {
            const pageBottom = window.innerHeight; // 获取视口的高度
            snowflakes.forEach(flake => {
                flake.y += Math.pow(flake.density, 2) * 1.5; // 调整下落速度
                flake.x += flake.xMovement * 1.2 + Math.cos(flake.y * 0.05) * 3; // 调整动态
                // 当雪花超出视口底部时，重新在顶部生成
                if (flake.y > pageBottom) {
                    flake.x = Math.random() * canvas.width;
                    flake.y = -flake.radius;
                }
            });
        };

        const intervalId = setInterval(drawSnowflakes, 30);

        return () => {
            clearInterval(intervalId);
            window.removeEventListener('resize', setCanvasSize);
        };
    }, []);

    return <canvas ref={canvasRef} style={{ position: 'fixed', top: 0, left: 0, pointerEvents: 'none', zIndex: 1 }} />;
};

export default SnowCanvas;
