
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Zap, Github, Linkedin, Sparkles, Layout, ChevronRight, ArrowRight, Code2, BookOpen } from 'lucide-react';
import { PROJECTS, BLOG_POSTS, SOCIAL_LINKS, TECH_STACK } from '../constants';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] }
};

const stagger = {
  animate: { transition: { staggerChildren: 0.1 } }
};

export const Home = () => {
  const navigate = useNavigate();
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }} 
      animate={{ opacity: 1, y: 0 }} 
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="pb-32"
    >
      {/* Hero Section */}
      <section className="pt-48 pb-16 max-w-5xl mx-auto px-4">
        <motion.div 
          initial="initial" animate="animate" variants={stagger}
          className="flex flex-col items-center text-center space-y-10"
        >
          <motion.div variants={fadeInUp} className="flex items-center space-x-2 px-4 py-1.5 rounded-full bg-blue-50 dark:bg-blue-400/10 border border-blue-100 dark:border-blue-400/20 text-blue-600 dark:text-blue-400 text-xs font-bold uppercase tracking-widest">
            <Zap size={14} fill="currentColor" />
            <span>System Status: Online</span>
          </motion.div>
          
          <motion.h1 
            variants={fadeInUp}
            className="text-6xl md:text-8xl font-black tracking-tighter leading-none"
          >
            Engineering <br />
            <span className="text-gradient">Autonomy.</span>
          </motion.h1>

          <motion.div 
            variants={fadeInUp}
            className="max-w-xl text-lg md:text-xl text-slate-500 dark:text-slate-400 font-medium leading-relaxed"
          >
            Ibrahim Ahmed — AI Engineer architecting agentic systems, robust MLOps pipelines, and scalable intelligence.
          </motion.div>

          <motion.div variants={fadeInUp} className="flex flex-wrap justify-center gap-4">
            <Link to="/projects" className="px-8 py-4 bg-blue-600 text-white rounded-full font-bold shadow-xl shadow-blue-600/25 hover:scale-105 transition-transform">
              View Work
            </Link>
            <div className="flex space-x-2">
              {SOCIAL_LINKS.map(link => (
                <a 
                  key={link.platform} 
                  href={link.url} 
                  target="_blank" 
                  className="p-4 rounded-full bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 hover:border-blue-500 transition-all"
                >
                  {link.platform === 'GitHub' && <Github size={20} />}
                  {link.platform === 'LinkedIn' && <Linkedin size={20} />}
                  {link.platform === 'HuggingFace' && <Sparkles size={20} />}
                </a>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </section>
      
      {/* Projects Preview */}
      <section className="max-w-5xl mx-auto px-4 mt-24">
        <div className="flex items-center justify-between mb-12">
          <div className="flex items-center space-x-4">
            <Layout className="text-blue-600" size={24} />
            <h2 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white">Featured Work</h2>
          </div>
          <Link to="/projects" className="text-sm font-bold text-blue-600 flex items-center group">
            All Projects <ChevronRight size={16} className="ml-1 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {PROJECTS.map((project) => (
            <motion.div 
              key={project.id}
              onClick={() => navigate(`/projects/${project.id}`)}
              initial="initial" whileInView="animate" variants={fadeInUp} viewport={{ once: true }}
              className="p-6 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2rem] hover:border-blue-500/50 transition-all group cursor-pointer flex flex-col"
            >
              <div className="aspect-square rounded-2xl overflow-hidden mb-6 flex-shrink-0">
                <img src={project.imageUrl} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" alt={project.title} />
              </div>
              <div className="flex gap-1.5 mb-3 flex-wrap">
                {project.tags.slice(0, 2).map(tag => (
                  <span key={tag} className="text-[9px] font-black uppercase tracking-widest text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-400/10 px-2 py-1 rounded">
                    {tag}
                  </span>
                ))}
              </div>
              <h4 className="text-lg font-bold mb-3 mono truncate text-slate-900 dark:text-white">{project.title}</h4>
              <p className="text-slate-500 dark:text-slate-400 text-sm line-clamp-2 mb-6 flex-grow">
                {project.description}
              </p>
              <div className="flex items-center text-[10px] font-black uppercase tracking-widest text-blue-600">
                View Specs <ArrowRight size={14} className="ml-2 group-hover:translate-x-1 transition-transform" />
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Stack Section */}
      <section className="max-w-5xl mx-auto px-4 mt-32">
        <div className="flex items-center space-x-4 mb-12">
            <Code2 className="text-blue-600" size={24} />
            <h2 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white">Core Stack</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {TECH_STACK.map(group => (
              <div key={group.category} className="p-8 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2rem]">
                <h5 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-6">{group.category}</h5>
                <div className="flex flex-col space-y-3">
                  {group.skills.map(skill => (
                    <span key={skill} className="text-sm font-bold text-slate-700 dark:text-slate-300">{skill}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
      </section>

      {/* Blog Preview */}
      <section className="max-w-5xl mx-auto px-4 mt-32">
        <div className="flex items-center justify-between mb-12">
          <div className="flex items-center space-x-4">
            <BookOpen className="text-blue-600" size={24} />
            <h2 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white">Recent Signals</h2>
          </div>
          <Link to="/blog" className="text-sm font-bold text-blue-600 flex items-center group">
            Full Blog <ChevronRight size={16} className="ml-1 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {BLOG_POSTS.slice(0, 3).map(post => (
            <motion.article 
              key={post.id} 
              onClick={() => navigate(`/blog/${post.id}`)}
              initial="initial" whileInView="animate" variants={fadeInUp} viewport={{ once: true }}
              className="p-8 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2rem] hover:border-blue-500/50 transition-all flex flex-col group cursor-pointer"
            >
              <div className="flex items-center space-x-3 mb-6 text-[10px] font-black uppercase tracking-widest text-slate-400">
                <span>{post.date}</span>
                <div className="w-1 h-1 rounded-full bg-slate-400" />
                <span className="text-blue-600">{post.category}</span>
              </div>
              <h3 className="text-xl font-bold mb-4 group-hover:text-blue-600 transition-colors leading-tight text-slate-900 dark:text-white">{post.title}</h3>
              <p className="text-slate-500 dark:text-slate-400 leading-relaxed text-sm mb-8 flex-grow">
                {post.excerpt}
              </p>
              <div className="pt-6 border-t border-slate-100 dark:border-white/5 flex items-center text-xs font-black uppercase tracking-widest text-blue-600">
                Read Signal <ArrowRight size={14} className="ml-2 group-hover:translate-x-1 transition-transform" />
              </div>
            </motion.article>
          ))}
        </div>
      </section>
    </motion.div>
  );
};
