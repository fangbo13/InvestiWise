import React from 'react';
import './Navbar.css'; // 导入组件的CSS样式
import { Link } from 'react-router-dom';


class Navbar extends React.Component {
  render() {
    return (
      <>
        {/* Logo 容器 */}
        <div className="logo-container">
          <Link to="/">
              <img src="/img/Logo.png" alt="header-Logo" className="logo" />
          </Link>
      </div>

        {/* 导航条 */}
        <nav className="navbar navbar-expand-lg">
          <div className="collapse navbar-collapse" id="navbarText">
            {/* 导航链接 */}
            <ul className="navbar-nav ml-auto line">
              <li className="nav-item">
                <a className="nav-link active" href="#home">Home</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#about">About</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#services">Services</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#tutorials">Tutorials</a>
              </li> 
            </ul>
          </div>  
        </nav>
      </>
    );
  }
}

export default Navbar;
