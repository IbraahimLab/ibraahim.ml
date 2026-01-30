
import React from 'react';
import { Link } from 'react-router-dom';
import { Mail, Phone, Github, Linkedin, Sparkles } from 'lucide-react';
import { SOCIAL_LINKS } from '../constants';

export const Footer = () => (
  <footer className="pt-32 pb-16 px-4 bg-slate-50 dark:bg-white/[0.02]">
    <div className="max-w-5xl mx-auto">
      <div className="flex flex-col md:flex-row justify-between items-start gap-12 mb-20">
        <div className="max-w-sm">
          <Link to="/" className="flex items-center space-x-2 mb-6">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold">I</div>
            <span className="font-bold text-xl">ibrahim.ml</span>
          </Link>
          <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed mb-8">
            Architecting agentic systems and scalable intelligence. Open for collaborative research.
          </p>
          <div className="space-y-2">
            <a href="mailto:ibraakadarba.12@gmail.com" className="flex items-center text-sm text-slate-400 hover:text-blue-600 transition-colors">
              <Mail size={14} className="mr-3" /> ibraakadarba.12@gmail.com
            </a>
            <div className="flex items-center text-sm text-slate-400">
              <Phone size={14} className="mr-3" /> +252-61-2537593
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-16">
          <div className="space-y-4">
            <h5 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Navigate</h5>
            <div className="flex flex-col space-y-2 text-sm font-semibold">
              <Link to="/" className="hover:text-blue-600 transition-colors">Home</Link>
              <Link to="/projects" className="hover:text-blue-600 transition-colors">Deployments</Link>
              <Link to="/blog" className="hover:text-blue-600 transition-colors">Signals</Link>
              <Link to="/tools" className="hover:text-blue-600 transition-colors">Library</Link>
            </div>
          </div>
          <div className="space-y-4">
            <h5 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Presence</h5>
            <div className="flex flex-col space-y-2 text-sm font-semibold">
              <a href="https://github.com/IbraahimLab" className="hover:text-blue-600 transition-colors">GitHub</a>
              <a href="https://huggingface.co/IbraahimLab" className="hover:text-blue-600 transition-colors">HuggingFace</a>
              <a href="https://linkedin.com/in/ibraahimahmed" className="hover:text-blue-600 transition-colors">LinkedIn</a>
            </div>
          </div>
        </div>
      </div>
      <div className="pt-8 border-t border-slate-200 dark:border-white/10 flex flex-col md:flex-row justify-between items-center gap-4 text-[10px] font-black uppercase tracking-widest text-slate-400">
        <span>© {new Date().getFullYear()} Ibrahim Ahmed Mohamud</span>
        <span>Built with Precision</span>
      </div>
    </div>
  </footer>
);
