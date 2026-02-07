
import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Sun, Moon, X, Menu } from 'lucide-react';
import { NavPill } from './NavPill';

export const Navbar = ({ darkMode, setDarkMode }: { darkMode: boolean, setDarkMode: (v: boolean) => void }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const links = [
    { name: 'Studio', path: '/' },
    { name: 'Deployments', path: '/projects' },
    { name: 'Signals', path: '/blog' },
    { name: 'Library', path: '/library' },
    { name: 'Tools', path: '/tools' },
  ];

  return (
    <nav className={`fixed top-0 left-0 w-full z-50 transition-all duration-500 pt-6 px-4`}>
      <div className={`max-w-5xl mx-auto flex items-center justify-between h-16 px-6 rounded-full border transition-all duration-500 ${
        scrolled ? 'bg-white/80 dark:bg-dark/80 backdrop-blur-xl border-slate-200 dark:border-white/10 shadow-2xl' : 'bg-transparent border-transparent'
      }`}>
        <Link to="/" className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white">
            <Cpu size={18} />
          </div>
          <span className="font-bold tracking-tight text-lg">ibrahim.ml</span>
        </Link>

        <div className="hidden md:flex items-center space-x-2">
          {links.map(link => (
            <NavPill key={link.path} to={link.path} active={location.pathname === link.path}>
              {link.name}
            </NavPill>
          ))}
          <div className="h-4 w-[1px] bg-slate-200 dark:bg-white/10 mx-4" />
          <button 
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-white/5 transition-colors"
          >
            {darkMode ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} className="text-blue-600" />}
          </button>
        </div>

        <button className="md:hidden" onClick={() => setMobileOpen(!mobileOpen)}>
          {mobileOpen ? <X /> : <Menu />}
        </button>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden absolute top-24 left-4 right-4 bg-white dark:bg-dark border border-slate-200 dark:border-white/10 rounded-3xl p-6 shadow-2xl"
          >
            <div className="flex flex-col space-y-4">
              {links.map(link => (
                <Link 
                  key={link.path} 
                  to={link.path} 
                  onClick={() => setMobileOpen(false)}
                  className="text-lg font-semibold"
                >
                  {link.name}
                </Link>
              ))}
              <div className="pt-4 border-t border-slate-100 dark:border-white/10">
                <button onClick={() => setDarkMode(!darkMode)} className="flex items-center space-x-2">
                  {darkMode ? <Sun size={20} /> : <Moon size={20} />}
                  <span>Appearance</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};
