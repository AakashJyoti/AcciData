import { createRoot } from 'react-dom/client'
import { StrictMode } from 'react'
import App from './App.tsx'
import './index.css'

// Remove the script tag that loads the React DevTools extension
// This helps prevent the "message port closed before response" error
const removeDevToolsScript = () => {
  const scripts = document.getElementsByTagName('script');
  for (let i = 0; i < scripts.length; i++) {
    const script = scripts[i];
    if (script.src && script.src.includes('gptengineer.js')) {
      script.parentNode?.removeChild(script);
      break;
    }
  }
};

// Call the function to remove the script
removeDevToolsScript();

// Use StrictMode to help catch potential issues
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
