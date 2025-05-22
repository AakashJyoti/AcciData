
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  const scrollToSection = (sectionId: string) => {
    // Only scroll if we're on the home page
    if (isHomePage) {
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      // If not on home page, navigate to home page with hash
      window.location.href = `/#${sectionId}`;
    }
  };

  return (
    <nav className="w-full py-4 bg-white shadow-sm">
      <div className="container mx-auto flex items-center justify-between px-4">
        <div className="flex items-center">
          <Link to="/" className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded bg-brand-600 flex items-center justify-center">
              <span className="text-white font-bold">A</span>
            </div>
            <span className="font-bold text-xl text-brand-800">AcciData</span>
          </Link>
        </div>

        <div className="hidden md:flex space-x-8 text-sm font-medium">
          <Link to="/" className="text-gray-700 hover:text-brand-600 transition-colors">
            Home
          </Link>
          {/* <Link to="/voice-agent" className="text-gray-700 hover:text-brand-600 transition-colors">
            Voice Assistant
          </Link> */}
          <Link to="/chat" className="text-gray-700 hover:text-brand-600 transition-colors">
            Chat
          </Link>
          <button
            onClick={() => scrollToSection('features')}
            className="text-gray-700 hover:text-brand-600 transition-colors"
          >
            Features
          </button>
          <button
            onClick={() => scrollToSection('how-it-works')}
            className="text-gray-700 hover:text-brand-600 transition-colors"
          >
            How It Works
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
