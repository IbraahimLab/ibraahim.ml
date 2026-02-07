import blogData from './content/blog.json';
import projectsData from './content/projects.json';
import toolsData from './content/tools.json';
import { type BlogPost, type Certification, type Project, type ResourceTool, type SkillGroup, type SocialLink } from './types';

export const SOCIAL_LINKS: SocialLink[] = [
  { platform: 'GitHub', url: 'https://github.com/IbraahimLab', icon: 'github' },
  { platform: 'LinkedIn', url: 'https://www.linkedin.com/in/ibraahimahmed/', icon: 'linkedin' },
  { platform: 'HuggingFace', url: 'https://huggingface.co/IbraahimLab', icon: 'face' },
];

export const PROJECTS: Project[] = projectsData as Project[];

export const TECH_STACK: SkillGroup[] = [
  { category: 'Core', skills: ['Python', 'SQL', 'Bash'] },
  { category: 'Intelligence', skills: ['PyTorch', 'LangChain', 'LangGraph', 'Transformers'] },
  { category: 'Infrastructure', skills: ['Docker', 'AWS', 'Linux/WSL', 'Redis'] },
  { category: 'Ops', skills: ['MLflow', 'DVC', 'LangSmith', 'GitHub Actions'] }
];

export const CERTIFICATIONS: Certification[] = [
  { name: 'Machine Learning Specialization', issuer: 'DeepLearning.AI' },
  { name: 'Managing Machine Learning Projects', issuer: 'Duke University' },
  { name: 'Machine Learning Operations BootCamp', issuer: 'Krish AI Technologies' }
];

export const BLOG_POSTS: BlogPost[] = blogData as BlogPost[];

export const RESOURCE_TOOLS: ResourceTool[] = toolsData as ResourceTool[];
