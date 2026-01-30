
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { PROJECTS } from '../constants';

export const Projects = () => {
  const navigate = useNavigate();
  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      exit={{ opacity: 0, y: -20 }}
      className="pt-48 max-w-5xl mx-auto px-4 pb-32"
    >
      <div className="max-w-2xl mb-24">
        <h1 className="text-5xl font-black mb-6 text-slate-900 dark:text-white">Deployments</h1>
        <p className="text-xl text-slate-500 font-medium">An iterative collection of research implementation and production-grade MLOps pipelines.</p>
      </div>
      <div className="space-y-12">
        {PROJECTS.map((p, i) => (
          <motion.div 
            key={p.id} 
            onClick={() => navigate(`/projects/${p.id}`)}
            className={`flex flex-col md:flex-row gap-12 items-center p-8 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2.5rem] hover:border-blue-500/50 transition-all cursor-pointer group ${i % 2 !== 0 ? 'md:flex-row-reverse' : ''}`}
          >
            <div className="flex-1 w-full rounded-2xl overflow-hidden shadow-2xl flex-shrink-0">
              <img src={p.imageUrl} className="w-full aspect-video object-cover group-hover:scale-105 transition-transform duration-700" alt={p.title} />
            </div>
            <div className="flex-1 space-y-6">
              <div className="flex gap-2">
                {p.tags.map(t => <span key={t} className="px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-400/10 text-blue-600 text-[10px] font-black uppercase tracking-widest">{t}</span>)}
              </div>
              <h3 className="text-3xl font-bold mono text-slate-900 dark:text-white">{p.title}</h3>
              <p className="text-lg text-slate-500 dark:text-slate-400 leading-relaxed line-clamp-3">{p.description}</p>
              <div className="inline-flex items-center space-x-2 text-blue-600 font-bold group">
                <span>Detailed Specs</span>
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};
