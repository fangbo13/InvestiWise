import React, { useEffect, useState } from 'react';
import './HomePage.css'; // 确保导入 CSS 文件

function HomePage() {
    const [settings, setSettings] = useState({ title: '', backgroundImage: '' });

    useEffect(() => {
        fetch('http://127.0.0.1:8000/api/home/')
            .then(response => response.json())
            .then(data => {
                setSettings({
                    title: data.heading,
                    backgroundImage: `http://127.0.0.1:8000${data.home_background}`
                });
            });
    }, []);

    const backgroundStyle = {
        backgroundImage: `url(${settings.backgroundImage})`,
    };

    return (
        <div className="homepage-container" style={backgroundStyle}>
            <div className="title-container">
                <h1 className="holiday-title">{settings.title}</h1>
            </div>
        </div>
    );
}

export default HomePage;
