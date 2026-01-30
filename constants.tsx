
import { Project, BlogPost, ResourceTool, ToolCategory, SocialLink, Certification, SkillGroup } from './types';

export const SOCIAL_LINKS: SocialLink[] = [
  { platform: 'GitHub', url: 'https://github.com/IbraahimLab', icon: 'github' },
  { platform: 'LinkedIn', url: 'https://www.linkedin.com/in/ibraahimahmed/', icon: 'linkedin' },
  { platform: 'HuggingFace', url: 'https://huggingface.co/IbraahimLab', icon: 'face' },
];

export const PROJECTS: Project[] = [
  {
    id: 'rag-expert-pipeline',
    title: 'rag-expert-pipeline',
    description: 'A specialized retrieval system architected to handle complex technical documentation for LLM fine-tuning. It utilizes a custom Docling extraction engine to parse multi-modal PDFs, maintains a high-precision vector index in ChromaDB, and includes an automated MLOps pipeline for AWS deployment via GitHub Actions.',
    tags: ['FastAPI', 'LangChain', 'AWS', 'Docker', 'MLOps'],
    githubUrl: 'https://github.com/IbraahimLab',
    imageUrl: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=800'
  },
  {
    id: 'vision-ml-skin-classifier',
    title: 'vision-ml-skin-classifier',
    description: 'An end-to-end medical vision pipeline focused on dermatological assessment. The project implements a configuration-driven training approach using YAML, incorporates DVC for data/model versioning, and tracks hyperparameter experiments with MLflow. Deployment is handled via a lightweight Flask inference service containerized with Docker.',
    tags: ['PyTorch', 'MLflow', 'DVC', 'CV'],
    githubUrl: 'https://github.com/IbraahimLab',
    imageUrl: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80&w=800'
  },
  {
    id: 'agentic-research-workflow',
    title: 'agentic-research-workflow',
    description: 'A sophisticated research agent capable of autonomous tool-augmented reasoning. By utilizing LangGraph, the agent can loop through multiple search phases—querying arXiv for papers, Serper for web context, and local PDF indices—before synthesizing a final response. Includes observability via LangSmith and RAGAS evaluation metrics.',
    tags: ['LangGraph', 'ChromaDB', 'Agentic AI', 'RAG'],
    githubUrl: 'https://github.com/IbraahimLab',
    imageUrl: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?auto=format&fit=crop&q=80&w=800'
  }
];

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

export const BLOG_POSTS: BlogPost[] = [
  {
    id: '1',
    title: 'Beyond the Prompt: Architecting Agentic RAG',
    excerpt: 'The transition from simple semantic similarity to complex, state-driven reasoning loops. This post dives into LangGraph implementation details for long-running research tasks and error-correction in autonomous agents.',
    content: `
      In the rapidly evolving landscape of Large Language Models (LLMs), basic Retrieval-Augmented Generation (RAG) is quickly becoming table stakes. While vector search provides a significant boost to model context, it often fails when faced with multi-step reasoning or dynamic information retrieval needs.

      ### Enter Agentic RAG
      
      Agentic RAG represents a paradigm shift from static pipelines to dynamic reasoning loops. Instead of a linear sequence of retrieval -> generation, we empower the model with a set of tools and a state management system. 

      ### Why LangGraph?
      
      LangGraph allows us to define the "cognitive architecture" of an agent using a directed graph. This is crucial for:
      1. **Persistence**: Maintaining state across multiple retrieval attempts.
      2. **Cycles**: Allowing the agent to "re-think" if the initial retrieval was insufficient.
      3. **Fine-grained Control**: Explicitly defining transitions between different search strategies.

      In my latest implementation, I've utilized LangGraph to coordinate between local PDF parsing (via Docling) and external research APIs (arXiv and Serper). This creates a truly autonomous researcher that doesn't just answer—it investigates.
    `,
    date: 'Feb 02, 2024',
    readTime: '10 min',
    category: 'Research'
  },
  {
    id: '2',
    title: 'MLOps at Scale: DVC & MLflow Integration',
    excerpt: 'Treating machine learning as a disciplined engineering practice. A guide on how to link data version control with experiment tracking to ensure reproducible research across heterogeneous cloud environments.',
    content: `
      Reproducibility is the bedrock of science, yet it remains one of the greatest challenges in machine learning engineering. "It works on my local machine" is a dangerous phrase when dealing with gigabytes of data and specific GPU driver versions.

      ### The MLOps Trinity
      
      To achieve true scalability, we need to treat three distinct artifacts with equal rigor:
      1. **Code**: Versioned in Git.
      2. **Data**: Versioned in DVC (Data Version Control).
      3. **Experiments**: Logged in MLflow.

      ### DVC as the Bridge
      
      DVC allows us to version datasets and model weights without bloating our Git repositories. By storing meta-pointers in Git, we can "checkout" a specific dataset version just as easily as we checkout a code branch.

      ### Tracking with MLflow
      
      Integration with MLflow ensures that every training run is captured. Parameters, metrics, and even model artifacts are logged automatically. In my projects, I use MLflow to compare CNN backbones—quickly identifying which architecture yields the best multi-label classification accuracy for dermatological images.
    `,
    date: 'Jan 12, 2024',
    readTime: '7 min',
    category: 'Engineering'
  },
  {
    id: '3',
    title: 'Automating the Edge: CI/CD for Embedded AI',
    excerpt: 'How to use GitHub Actions and ECR to push model updates to edge devices securely. We discuss containerization strategies that minimize footprint while maintaining performance.',
    content: `
      Deploying AI to the cloud is one thing; deploying to the edge is another entirely. Resource constraints, intermittent connectivity, and security requirements make embedded AI a unique challenge.

      ### The Pipeline
      
      A robust edge deployment pipeline follows these steps:
      1. **Standardization**: Containerizing the inference service using Docker.
      2. **Automation**: Using GitHub Actions to trigger builds on every push.
      3. **Registry**: Pushing optimized images to AWS ECR (Elastic Container Registry).

      ### Optimizing for Footprint
      
      When working with edge devices like AWS EC2 T-series or local IoT gateways, every megabyte counts. I focus on multi-stage builds and specialized inference engines like vLLM or ONNX Runtime to ensure that our models remain responsive without over-consuming memory.
    `,
    date: 'Dec 05, 2023',
    readTime: '5 min',
    category: 'Infrastructure'
  }
];

export const RESOURCE_TOOLS: ResourceTool[] = [
  // --- TRAINING & INFRA ---
  { id: 'train1', name: 'Megatron-LM', description: 'NVIDIA\'s high-performance framework for training massive transformer models using model parallelism.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/NVIDIA/Megatron-LM', tags: ['Pre-training', 'Model-Parallel', 'NVIDIA'] },
  { id: 'train2', name: 'DeepSpeed', description: 'Microsoft\'s optimization library for distributed training, featuring ZeRO (Zero Redundancy Optimizer).', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/microsoft/DeepSpeed', tags: ['Optimization', 'Distributed', 'ZeRO'] },
  { id: 'train3', name: 'Colossal-AI', description: 'Unified distributed training system for large-scale AI models, integrating multiple parallelism strategies.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/hpcaitech/ColossalAI', tags: ['Parallelism', 'Efficiency'] },
  { id: 'train4', name: 'PyTorch FSDP', description: 'Fully Sharded Data Parallelism for scaling model training with lower memory footprints natively in PyTorch.', category: ToolCategory.TRAINING_INFRA, url: 'https://pytorch.org/docs/stable/fsdp.html', tags: ['Native', 'FSDP', 'Scaling'] },
  { id: 'train5', name: 'Torchtitan', description: 'Meta\'s clean-slate, PyTorch-native implementation of LLM training used for research and scale.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/pytorch/torchtitan', tags: ['Meta', 'PyTorch-Native'] },
  { id: 'train6', name: 'GPT-NeoX', description: 'EleutherAI\'s library for training large-scale language models on GPUs, powered by DeepSpeed.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/EleutherAI/gpt-neox', tags: ['EleutherAI', 'Training'] },
  { id: 'train7', name: 'Axolotl', description: 'A tool for making LLM fine-tuning easier and more efficient across various architectures.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/OpenAccess-AI-Collective/axolotl', tags: ['Fine-tuning', 'Configs'] },
  { id: 'train8', name: 'Unsloth', description: 'Highly optimized kernels that make LLM fine-tuning 2x faster and use 70% less memory.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/unslothai/unsloth', tags: ['Optimization', 'Fast-Fine-tuning'] },
  { id: 'train9', name: 'NanoGPT', description: 'Andrej Karpathy\'s repository for training medium-sized GPTs; excellent for educational pre-training.', category: ToolCategory.TRAINING_INFRA, url: 'https://github.com/karpathy/nanoGPT', tags: ['Educational', 'Clean-Code'] },

  // --- LLM FRAMEWORKS ---
  { id: 't1', name: 'LangChain', description: 'Comprehensive framework for developing applications powered by LLMs.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://langchain.com', tags: ['Orchestration', 'RAG'] },
  { id: 't4', name: 'LlamaIndex', description: 'Data framework for LLM applications to connect custom data sources.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://www.llamaindex.ai', tags: ['Data', 'Indexing'] },
  { id: 't5', name: 'Haystack', description: 'Open-source NLP framework to build search systems and RAG pipelines.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://haystack.deepset.ai', tags: ['Search', 'NLP'] },
  { id: 't6', name: 'DSPy', description: 'Framework for programming—rather than prompting—language models.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://github.com/stanfordnlp/dspy', tags: ['Optimization', 'Logic'] },
  { id: 't7', name: 'AutoGen', description: 'Framework for building multi-agent systems that converse to solve tasks.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://microsoft.github.io/autogen', tags: ['Agents', 'Multi-Agent'] },
  { id: 't8', name: 'CrewAI', description: 'Framework for orchestrating role-playing, autonomous AI agents.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://www.crewai.com', tags: ['Agents', 'Collaboration'] },
  { id: 't9', name: 'LangGraph', description: 'Library for building stateful, multi-actor applications with LLMs.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://langchain-ai.github.io/langgraph', tags: ['State-Machine', 'Cycles'] },
  { id: 't10', name: 'Guidance', description: 'A guidance language for controlling large language models.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://github.com/guidance-ai/guidance', tags: ['Constrained-Output', 'Template'] },
  { id: 't11', name: 'Outlines', description: 'Guided text generation for LLMs with regex and JSON schema support.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://github.com/outlines-dev/outlines', tags: ['JSON', 'Structured'] },
  { id: 't12', name: 'Instructor', description: 'Structured extraction from LLMs using Pydantic models.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://github.com/jxnl/instructor', tags: ['Pydantic', 'Extraction'] },
  { id: 't13', name: 'Semantic Kernel', description: 'SDK that integrates LLMs with conventional programming languages.', category: ToolCategory.LLM_FRAMEWORK, url: 'https://github.com/microsoft/semantic-kernel', tags: ['SDK', 'Enterprise'] },

  // --- OPEN SOURCE MODELS ---
  { id: 't2', name: 'vLLM', description: 'High-throughput and memory-efficient inference engine for LLMs.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/vllm-project/vllm', tags: ['Inference', 'Cuda'] },
  { id: 't14', name: 'Llama 3.1', description: 'Meta\'s state-of-the-art open source large language model.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://llama.meta.com', tags: ['Base-Model', 'Open-Weights'] },
  { id: 't15', name: 'Mistral NeMo', description: '12B parameter model built in collaboration with NVIDIA.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://mistral.ai', tags: ['Efficiency', 'Mistral'] },
  { id: 't16', name: 'DeepSeek-V2', description: 'Strong MoE (Mixture of Experts) model with excellent coding abilities.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/deepseek-ai/DeepSeek-V2', tags: ['MoE', 'Coding'] },
  { id: 't17', name: 'Gemma 2', description: 'Google\'s lightweight, state-of-the-art open models built from Gemini technology.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://ai.google.dev/gemma', tags: ['Google', 'Lightweight'] },
  { id: 't18', name: 'Phi-3.5', description: 'Microsoft\'s small language models with massive reasoning capabilities.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/microsoft/Phi-3CookBook', tags: ['SLM', 'Edge'] },
  { id: 't19', name: 'Qwen 2.5', description: 'Alibaba\'s latest series of LLMs with strong multilingual and math skills.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/QwenLM/Qwen2.5', tags: ['Multilingual', 'Math'] },
  { id: 't20', name: 'Grok-1', description: 'xAI\'s 314 billion parameter Mixture-of-Experts model.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/xai-org/grok-1', tags: ['Massive', 'xAI'] },
  { id: 't21', name: 'Falcon 180B', description: 'TII\'s massive model, one of the largest open-access weights.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://falconllm.tii.ae', tags: ['UAE', 'Scale'] },
  { id: 't22', name: 'StarCoder2', description: 'Open LLMs for code, trained on The Stack v2.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/bigcode-project/starcoder2', tags: ['Code', 'Training'] },
  { id: 't23', name: 'Stable Diffusion XL', description: 'The gold standard for high-quality open-source image generation.', category: ToolCategory.OPEN_SOURCE_MODELS, url: 'https://github.com/Stability-AI/generative-models', tags: ['Vision', 'Diffusion'] },

  // --- PLAYLISTS ---
  { id: 'p1', name: 'Machine Learning Specialization', description: 'Andrew Ng\'s legendary course, recently updated with modern ML techniques.', category: ToolCategory.PLAYLIST, url: 'https://www.coursera.org/specializations/machine-learning-introduction', tags: ['DeepLearning.AI', 'Basics'] },
  { id: 'p2', name: 'Neural Networks: Zero to Hero', description: 'Andrej Karpathy\'s elite series on building neural networks from scratch.', category: ToolCategory.PLAYLIST, url: 'https://karpathy.ai/zero-to-hero.html', tags: ['Karpathy', 'Implementation'] },
  { id: 'p3', name: 'Practical Deep Learning for Coders', description: 'Fast.ai\'s top-down approach to mastering deep learning applications.', category: ToolCategory.PLAYLIST, url: 'https://course.fast.ai/', tags: ['Fast.ai', 'Practical'] },
  { id: 'p4', name: 'CS231n: Vision Deep Learning', description: 'Stanford\'s premier course on convolutional neural networks for visual recognition.', category: ToolCategory.PLAYLIST, url: 'http://cs231n.stanford.edu/', tags: ['Stanford', 'Computer Vision'] },
  { id: 'p5', name: 'CS224n: NLP with Deep Learning', description: 'Stanford\'s comprehensive guide to the latest in natural language processing.', category: ToolCategory.PLAYLIST, url: 'http://web.stanford.edu/class/cs224n/', tags: ['Stanford', 'NLP'] },
  { id: 'p6', name: 'MIT 6.S191: Deep Learning', description: 'Intensive introduction to deep learning algorithms and their applications.', category: ToolCategory.PLAYLIST, url: 'http://introtodeeplearning.com/', tags: ['MIT', 'Algorithms'] },

  // --- FUNDAMENTAL REASONING & LOGIC PAPERS ---
  { id: 'paper_ds_r1', name: 'DeepSeek-R1', description: 'Groundbreaking 2025 paper on incentivizing reasoning via pure Reinforcement Learning (RL), establishing new open-source benchmarks.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2501.12948', tags: ['DeepSeek', 'RL', 'Reasoning'] },
  { id: 'paper_cot', name: 'Chain-of-Thought Prompting', description: 'The foundational work that proved eliciting intermediate reasoning steps significantly improves LLM performance.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2201.11903', tags: ['CoT', 'Elicitation'] },
  { id: 'paper_zs_cot', name: 'Zero-Shot CoT', description: 'The famous "Let\'s think step by step" paper, demonstrating that reasoning is an emergent zero-shot capability.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2205.11916', tags: ['Zero-Shot', 'CoT'] },
  { id: 'paper_star', name: 'STaR: Self-Taught Reasoner', description: 'Bootstrapping reasoning by letting models generate and learn from their own correct reasoning traces.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2203.14465', tags: ['Self-Taught', 'Training'] },
  { id: 'paper_tot', name: 'Tree of Thoughts', description: 'A framework for deliberate problem solving by exploring and self-evaluating multiple reasoning branches.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2305.10601', tags: ['ToT', 'Search'] },
  { id: 'paper_bot', name: 'Buffer of Thoughts', description: 'Thought-augmented reasoning that uses a meta-buffer of "thought templates" to solve complex tasks efficiently.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2406.04271', tags: ['Efficiency', 'Logic'] },
  { id: 'paper_react', name: 'ReAct: Reasoning & Acting', description: 'Integrating reasoning traces and task-specific actions to create reliable autonomous agent loops.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2210.03629', tags: ['Agents', 'Tool-Use'] },
  { id: 'paper_prm', name: 'Let\'s Verify Step by Step', description: 'Introduction of Process-based Reward Models (PRMs) to reward logic steps instead of just final outcomes.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2305.20050', tags: ['PRM', 'Alignment'] },
  { id: 'paper_sc', name: 'Self-Consistency for CoT', description: 'A decoding strategy that samples multiple reasoning paths and selects the most consistent answer.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2203.11171', tags: ['Self-Consistency', 'Decoding'] },
  { id: 'paper_cove', name: 'Chain-of-Verification (CoVe)', description: 'Method for models to deliberate on their own responses to reduce hallucinations through self-correction.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2309.11495', tags: ['Hallucination', 'Verification'] },
  { id: 'paper_quiet_star', name: 'Quiet-STaR', description: 'Algorithm that allows models to reason "quietly" in the background of every token produced.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2403.09629', tags: ['Background-Reasoning', 'STaR'] },
  { id: 'paper_reflexion', name: 'Reflexion: Language Agents', description: 'Agentic framework using linguistic feedback to reinforce successful reasoning strategies.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/2303.11366', tags: ['Feedback', 'Reinforcement'] },

  // --- ARCHITECTURE & FOUNDATION PAPERS ---
  { id: 'paper1', name: 'Attention Is All You Need', description: 'The foundational paper that introduced the Transformer architecture.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/1706.03762', tags: ['Transformer', 'Must-Read'] },
  { id: 'paper2', name: 'ResNet: Deep Residual Learning', description: 'The paper that enabled training of extremely deep neural networks via skip connections.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/1512.03385', tags: ['Computer Vision', 'ResNet'] },
  { id: 'paper3', name: 'BERT: Pre-training Transformers', description: 'Introduced bidirectional pre-training for language understanding tasks.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/1810.04805', tags: ['NLP', 'BERT'] },
  { id: 'paper4', name: 'Adam Optimizer', description: 'The definitive paper on the most widely used optimization algorithm in DL.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/1412.6980', tags: ['Optimization', 'Adam'] },
  { id: 'paper5', name: 'YOLO: You Only Look Once', description: 'A revolutionary approach to real-time object detection.', category: ToolCategory.PAPERS, url: 'https://arxiv.org/abs/1506.02640', tags: ['Object Detection', 'Vision'] },

  // --- CLOUD PLATFORMS ---
  { id: 't24', name: 'Lambda Labs', description: 'GPU cloud with top-tier training and inference performance.', category: ToolCategory.ML_DL_CLOUD, url: 'https://lambdalabs.com', tags: ['GPU', 'H100'] },
  { id: 't25', name: 'RunPod', description: 'Rent GPUs for AI/ML workloads with easy Docker integration.', category: ToolCategory.ML_DL_CLOUD, url: 'https://www.runpod.io', tags: ['Serverless', 'On-Demand'] },
  { id: 't26', name: 'Together AI', description: 'The fastest cloud platform for fine-tuning and inference.', category: ToolCategory.ML_DL_CLOUD, url: 'https://www.together.ai', tags: ['API', 'Fine-Tuning'] },
  { id: 't27', name: 'Groq', description: 'LPU inference engine delivering unprecedented speed for LLMs.', category: ToolCategory.ML_DL_CLOUD, url: 'https://groq.com', tags: ['Inference', 'LPU'] },
  { id: 't28', name: 'Anyscale', description: 'Managed Ray platform for scaling AI and Python workloads.', category: ToolCategory.ML_DL_CLOUD, url: 'https://www.anyscale.com', tags: ['Ray', 'Compute'] },
  { id: 't29', name: 'Modal', description: 'The fastest way to run generative AI code in the cloud.', category: ToolCategory.ML_DL_CLOUD, url: 'https://modal.com', tags: ['Serverless', 'Pythonic'] },

  // --- DATA CURATION ---
  { id: 't3', name: 'DVC', description: 'Open-source Data Version Control for machine learning projects.', category: ToolCategory.DATA_CURATION, url: 'https://dvc.org', tags: ['Versioning', 'Git'] },
  { id: 't34', name: 'Unstructured.io', description: 'Open source library to ingest and preprocess unstructured data for RAG.', category: ToolCategory.DATA_CURATION, url: 'https://unstructured.io', tags: ['Preprocessing', 'PDF'] },
  { id: 't35', name: 'Label Studio', description: 'Multi-type data labeling and annotation tool for AI.', category: ToolCategory.DATA_CURATION, url: 'https://labelstud.io', tags: ['Annotation', 'Open-Source'] },
  { id: 't36', name: 'Cleanlab', description: 'Software to find and fix label errors in datasets automatically.', category: ToolCategory.DATA_CURATION, url: 'https://cleanlab.ai', tags: ['Quality', 'Cleaning'] },
  { id: 't40', name: 'LangSmith', description: 'Platform for debugging, testing, and monitoring LLM applications.', category: ToolCategory.DATA_CURATION, url: 'https://www.langchain.com/langsmith', tags: ['Observability', 'Traces'] },

  // --- COMMUNITIES ---
  { id: 't43', name: 'Hugging Face Discord', description: 'The main hub for modern AI research and implementation discussions.', category: ToolCategory.COMMUNITY, url: 'https://huggingface.co/join/discord', tags: ['Hub', 'Collaboration'] },
  { id: 't44', name: 'LocalLLaMA', description: 'Reddit community dedicated to running LLMs on consumer hardware.', category: ToolCategory.COMMUNITY, url: 'https://www.reddit.com/r/LocalLLaMA', tags: ['Edge', 'Consumer'] },
  { id: 't45', name: 'EleutherAI', description: 'Grass-roots non-profit AI research lab known for GPT-Neo/J.', category: ToolCategory.COMMUNITY, url: 'https://www.eleuther.ai', tags: ['Research', 'Open-Science'] },
  { id: 't46', name: 'MLOps.community', description: 'The definitive place to talk about machine learning in production.', category: ToolCategory.COMMUNITY, url: 'https://mlops.community', tags: ['Operations', 'Networking'] },
  { id: 't47', name: 'Latent Space', description: 'Newsletter and community focused on the "AI Engineer" transition.', category: ToolCategory.COMMUNITY, url: 'https://www.latent.space', tags: ['Learning', 'Trends'] }
];
