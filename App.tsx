
import React, { useState, useEffect } from 'react';
import { HashRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';

// Layout Components
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Background } from './components/Background';

// Page Components
import { Home } from './pages/Home';
import { Projects } from './pages/Projects';
import { ProjectDetail } from './pages/ProjectDetail';
import { Blog } from './pages/Blog';
import { BlogPost } from './pages/BlogPost';
import { Library } from './pages/Library';
import { Tools } from './pages/Tools';

const AnimatedRoutes = () => {
  const location = useLocation();
  
  return (
    // initial={false} prevents the entering animation on the very first load
    // which often causes glitches in single-page route mounting.
    <AnimatePresence mode="wait" initial={false}>
      <Routes location={location}>
        <Route path="/" element={<Home />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/projects/:id" element={<ProjectDetail />} />
        <Route path="/blog" element={<Blog />} />
        <Route path="/blog/:id" element={<BlogPost />} />
        <Route path="/library" element={<Library />} />
        <Route path="/tools" element={<Tools />} />
      </Routes>
    </AnimatePresence>
  );
};

const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      document.body.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
      document.body.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <div className="min-h-screen selection:bg-blue-600 selection:text-white transition-colors duration-500 bg-slate-50 dark:bg-dark flex flex-col">
      <Background />
      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      
      {/* min-h-[70vh] ensures the footer doesn't jump up when pages swap */}
      <main className="relative z-10 flex-grow w-full min-h-[70vh]">
        <AnimatedRoutes />
      </main>

      <Footer />
    </div>
  );
};

export default function AppWrapper() {
  return (
    <Router>
      <App />
    </Router>
  );
}
