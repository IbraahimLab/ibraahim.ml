
import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, MessageSquare, User, Send } from 'lucide-react';
import { BLOG_POSTS } from '../constants';

export const BlogPost = () => {
  const { id } = useParams<{ id: string }>();
  const post = BLOG_POSTS.find(p => p.id === id);

  if (!post) {
    return (
      <motion.div 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }}
        className="pt-48 pb-32 max-w-5xl mx-auto px-4 text-center"
      >
        <h1 className="text-4xl font-bold mb-4">Post not found</h1>
        <Link to="/blog" className="text-blue-600 font-bold flex items-center justify-center">
          <ArrowLeft size={18} className="mr-2" /> Back to Signals
        </Link>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }} 
      animate={{ opacity: 1, y: 0 }} 
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4 }}
      className="pt-48 max-w-4xl mx-auto px-4 pb-32"
    >
      <Link to="/blog" className="inline-flex items-center text-sm font-bold text-slate-400 hover:text-blue-600 transition-colors mb-12 group">
        <ArrowLeft size={16} className="mr-2 group-hover:-translate-x-1 transition-transform" /> Back to Signals
      </Link>

      <div className="mb-12">
        <div className="flex items-center space-x-4 mb-6 text-xs font-black uppercase tracking-widest text-slate-400">
          <span>{post.date}</span>
          <div className="w-1 h-1 rounded-full bg-slate-400" />
          <span className="text-blue-600">{post.category}</span>
          <div className="w-1 h-1 rounded-full bg-slate-400" />
          <span>{post.readTime}</span>
        </div>
        <h1 className="text-4xl md:text-6xl font-black mb-8 leading-tight text-slate-900 dark:text-white">{post.title}</h1>
        
        <div className="flex items-center space-x-4 p-6 bg-blue-50 dark:bg-blue-400/10 rounded-2xl border border-blue-100 dark:border-blue-400/20">
          <div className="w-12 h-12 rounded-full bg-blue-600 flex items-center justify-center text-white text-xl font-bold shadow-lg shadow-blue-600/20">
            I
          </div>
          <div>
            <h4 className="font-bold text-slate-900 dark:text-white">Ibrahim Ahmed Mohamud</h4>
            <p className="text-xs text-slate-500 font-medium">AI / ML Engineer & Technical Writer</p>
          </div>
        </div>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none mb-20 text-slate-600 dark:text-slate-300 leading-relaxed whitespace-pre-line">
        {post.content}
      </div>

      <section className="pt-16 border-t border-slate-200 dark:border-white/10">
        <div className="flex items-center space-x-4 mb-10">
          <MessageSquare className="text-blue-600" size={24} />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Comments (2)</h2>
        </div>

        <div className="space-y-8 mb-12">
          {[
            { user: "Alex Chen", time: "2 Days Ago", text: "Really great breakdown of the LangGraph state machine. I've been struggling with cycles in my agents, this cleared things up!" },
            { user: "Sarah Miller", time: "Yesterday", text: "Do you have any recommendations for RAGAS evaluation metrics when using Serper search?" }
          ].map((comment, i) => (
            <div key={i} className="flex space-x-4">
              <div className="w-10 h-10 rounded-full bg-slate-200 dark:bg-white/10 flex items-center justify-center flex-shrink-0">
                <User size={18} className="text-slate-400" />
              </div>
              <div className="flex-grow p-6 bg-slate-50 dark:bg-white/[0.02] rounded-2xl border border-slate-100 dark:border-white/[0.05]">
                <div className="flex justify-between items-center mb-2">
                  <h5 className="font-bold text-sm text-slate-900 dark:text-white">{comment.user}</h5>
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">{comment.time}</span>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
                  {comment.text}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="p-8 bg-white dark:bg-dark border border-slate-200 dark:border-white/10 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-none">
          <h4 className="font-bold mb-6 text-slate-900 dark:text-white">Join the Signal</h4>
          <div className="flex flex-col space-y-4">
            <textarea 
              placeholder="What are your thoughts?"
              className="w-full p-4 bg-slate-50 dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-xl focus:ring-2 focus:ring-blue-600 outline-none transition-all resize-none h-32 dark:text-white"
            />
            <button className="self-end px-8 py-3 bg-blue-600 text-white rounded-full font-bold flex items-center hover:scale-105 transition-transform">
              Post Comment <Send size={16} className="ml-2" />
            </button>
          </div>
        </div>
      </section>
    </motion.div>
  );
};
