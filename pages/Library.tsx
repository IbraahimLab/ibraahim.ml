
import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ExternalLink } from 'lucide-react';
import { RESOURCE_TOOLS } from '../constants';
import { ToolCategory } from '../types';

export const Library = () => {
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState<ToolCategory | 'All'>('All');

  const filtered = useMemo(() => RESOURCE_TOOLS.filter(t => {
    const m = t.name.toLowerCase().includes(search.toLowerCase()) || t.description.toLowerCase().includes(search.toLowerCase());
    const c = activeCategory === 'All' || t.category === activeCategory;
    return m && c;
  }), [search, activeCategory]);

  const categories = ['All', ...Object.values(ToolCategory)];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }} 
      animate={{ opacity: 1, y: 0 }} 
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="pt-48 max-w-5xl mx-auto px-4 pb-32"
    >
      <div className="max-w-xl mb-16">
        <h1 className="text-5xl font-black mb-6 text-slate-900 dark:text-white">Library.db</h1>
        <p className="text-lg text-slate-500 font-medium">Curated dependencies, frameworks, and foundational models for AI engineering.</p>
      </div>

      <div className="space-y-10 mb-12">
        <div className="relative max-w-2xl">
          <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
          <input 
            type="text" 
            placeholder="Search AI engineering resources..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-14 pr-6 py-5 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-3xl focus:ring-2 focus:ring-blue-600 outline-none shadow-xl shadow-slate-200/20 dark:shadow-none text-lg transition-all dark:text-white"
          />
        </div>
        
        <div className="flex flex-wrap gap-2">
          {categories.map(cat => (
            <button 
              key={cat} 
              onClick={() => setActiveCategory(cat as any)}
              className="relative px-6 py-2.5 rounded-full text-xs font-black uppercase tracking-widest transition-colors duration-300"
            >
              <span className={`relative z-10 ${activeCategory === cat ? 'text-white' : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'}`}>
                {cat}
              </span>
              {activeCategory === cat && (
                <motion.div 
                  layoutId="library-filter-pill"
                  className="absolute inset-0 bg-blue-600 rounded-full shadow-lg shadow-blue-600/30"
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                />
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <AnimatePresence mode="popLayout">
          {filtered.map(tool => (
            <motion.div 
              layout
              key={tool.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="p-8 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2rem] flex flex-col hover:border-blue-500 transition-all group"
            >
              <div className="flex justify-between items-start mb-4">
                <span className="text-[10px] font-black text-blue-600 uppercase tracking-widest">{tool.category}</span>
                <a href={tool.url} target="_blank" className="text-slate-400 hover:text-blue-600 transition-colors"><ExternalLink size={16} /></a>
              </div>
              <h4 className="text-lg font-bold mb-2 text-slate-900 dark:text-white">{tool.name}</h4>
              <p className="text-sm text-slate-500 mb-6 flex-grow leading-relaxed">{tool.description}</p>
              <div className="flex flex-wrap gap-2 mt-auto">
                {tool.tags.map(t => <span key={t} className="text-[9px] font-mono font-bold text-slate-400 bg-slate-100 dark:bg-white/5 px-2 py-0.5 rounded">#{t}</span>)}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};
