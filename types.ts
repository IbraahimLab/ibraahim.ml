
export interface Project {
  id: string;
  title: string;
  description: string;
  tags: string[];
  githubUrl?: string;
  demoUrl?: string;
  imageUrl: string;
}

export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  date: string;
  readTime: string;
  category: string;
}

export enum ToolCategory {
  LLM_FRAMEWORK = 'LLM Frameworks',
  TRAINING_INFRA = 'Training & Infra',
  ML_DL_CLOUD = 'Cloud Platforms',
  DATA_CURATION = 'Data Curation',
  OPEN_SOURCE_MODELS = 'OS Models',
  PLAYLIST = 'Playlists',
  PAPERS = 'Papers',
  COMMUNITY = 'Communities'
}

export interface ResourceTool {
  id: string;
  name: string;
  description: string;
  category: ToolCategory;
  url: string;
  tags: string[];
}

export interface SocialLink {
  platform: string;
  url: string;
  icon: string;
}

export interface Certification {
  name: string;
  issuer: string;
}

export interface SkillGroup {
  category: string;
  skills: string[];
}
