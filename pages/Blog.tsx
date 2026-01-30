
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ChevronRight } from 'lucide-react';
import { BLOG_POSTS } from '../constants';

export const Blog = () => {
  const navigate = useNavigate();
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }} 
      animate={{ opacity: 1, y: 0 }} 
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4 }}
      className="pt-48 max-w-3xl mx-auto px-4 pb-32"
    >
      <div className="mb-24">
        <h1 className="text-5xl font-black mb-6 text-slate-900 dark:text-white">Signals</h1>
        <p className="text-xl text-slate-500 font-medium">Observations on the rapid evolution of autonomous agents and machine intelligence.</p>
      </div>
      <div className="space-y-16">
        {BLOG_POSTS.map(post => (
          <article 
            key={post.id} 
            onClick={() => navigate(`/blog/${post.id}`)} 
            className="group cursor-pointer"
          >
            <div className="flex items-center space-x-4 mb-4 text-xs font-bold uppercase tracking-widest text-slate-400">
              <span>{post.date}</span>
              <div className="w-1 h-1 rounded-full bg-slate-400" />
              <span className="text-blue-600">{post.category}</span>
            </div>
            <h2 className="text-3xl font-bold mb-4 text-slate-900 dark:text-white group-hover:text-blue-600 transition-colors">{post.title}</h2>
            <p className="text-lg text-slate-500 leading-relaxed mb-6 line-clamp-3">{post.excerpt}</p>
            <div className="flex items-center text-sm font-black uppercase tracking-widest text-blue-600">
              Read Signal <ChevronRight size={16} className="ml-1 group-hover:translate-x-1 transition-transform" />
            </div>
          </article>
        ))}
      </div>
    </motion.div>
  );
};
