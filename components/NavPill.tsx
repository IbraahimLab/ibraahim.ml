
import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

export const NavPill: React.FC<{ to: string; children: React.ReactNode; active: boolean }> = ({ to, children, active }) => (
  <Link 
    to={to} 
    className={`relative px-5 py-2 text-sm font-medium transition-colors duration-300 ${
      active ? 'text-blue-600 dark:text-blue-400' : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
    }`}
  >
    {children}
    {active && (
      <motion.div 
        layoutId="active-pill"
        className="absolute inset-0 bg-blue-50 dark:bg-blue-400/10 rounded-full -z-10"
        transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
      />
    )}
  </Link>
);
