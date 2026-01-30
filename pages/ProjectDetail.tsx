
import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Github, Terminal, CircleDot, Zap } from 'lucide-react';
import { PROJECTS } from '../constants';

export const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const project = PROJECTS.find(p => p.id === id);

  if (!project) {
    return (
      <motion.div 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        exit={{ opacity: 0 }}
        className="pt-48 pb-32 max-w-5xl mx-auto px-4 text-center"
      >
        <h1 className="text-4xl font-bold mb-4">Project not found</h1>
        <Link to="/projects" className="text-blue-600 font-bold flex items-center justify-center">
          <ArrowLeft size={18} className="mr-2" /> Back to Deployments
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
      className="pt-48 max-w-5xl mx-auto px-4 pb-32"
    >
      <Link to="/projects" className="inline-flex items-center text-sm font-bold text-slate-400 hover:text-blue-600 transition-colors mb-12 group">
        <ArrowLeft size={16} className="mr-2 group-hover:-translate-x-1 transition-transform" /> Back to Deployments
      </Link>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 mb-24">
        <div>
          <div className="flex gap-2 mb-6">
            {project.tags.map(tag => (
              <span key={tag} className="text-[10px] font-black uppercase tracking-widest text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-400/10 px-3 py-1.5 rounded-lg">
                {tag}
              </span>
            ))}
          </div>
          <h1 className="text-5xl font-black mb-8 leading-tight mono text-slate-900 dark:text-white">{project.title}</h1>
          <p className="text-xl text-slate-500 dark:text-slate-400 leading-relaxed mb-10">
            {project.description}
          </p>
          <div className="flex gap-4">
            <a 
              href={project.githubUrl} 
              target="_blank" 
              className="px-8 py-4 bg-blue-600 text-white rounded-full font-bold shadow-xl shadow-blue-600/25 hover:scale-105 transition-transform flex items-center"
            >
              <Github size={18} className="mr-2" /> Source Implementation
            </a>
          </div>
        </div>
        <div className="rounded-[3rem] overflow-hidden border border-slate-200 dark:border-white/10 shadow-2xl">
          <img src={project.imageUrl} className="w-full h-full object-cover" alt={project.title} />
        </div>
      </div>

      <div className="space-y-16">
        <section>
          <h2 className="text-2xl font-bold mb-6 flex items-center text-slate-900 dark:text-white">
            <Terminal size={20} className="mr-3 text-blue-600" /> Technical Overview
          </h2>
          <div className="p-10 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2.5rem]">
            <p className="text-slate-500 dark:text-slate-400 leading-relaxed">
              This system was designed to address scalability and reliability in automated AI workflows. By leveraging {project.tags[0]} for core logic and {project.tags[1]} for orchestration, we achieved a modular architecture that supports rapid iteration. The deployment utilizes {project.tags[2]} to ensure high availability across globally distributed environments.
            </p>
          </div>
        </section>

        <section>
          <h2 className="text-2xl font-bold mb-6 flex items-center text-slate-900 dark:text-white">
             <CircleDot size={20} className="mr-3 text-blue-600" /> Key Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              "Automated MLOps lifecycle with GitHub Actions integration",
              "Production-ready inference endpoints with high throughput",
              "Comprehensive observability through LangSmith and custom logging",
              "Modular design pattern for seamless model upgrades"
            ].map((feature, i) => (
              <div key={i} className="flex items-start space-x-4 p-6 bg-slate-50 dark:bg-white/[0.02] rounded-2xl border border-slate-100 dark:border-white/[0.05]">
                <div className="w-6 h-6 rounded-full bg-blue-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Zap size={14} className="text-blue-600" />
                </div>
                <span className="text-slate-600 dark:text-slate-400 font-medium">{feature}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </motion.div>
  );
};
