
import React from 'react';

export const Background = () => (
  <div className="fixed inset-0 -z-50 overflow-hidden pointer-events-none opacity-40">
    <div className="absolute top-[-10%] right-[-10%] w-[60%] h-[60%] bg-blue-500/10 blur-[120px] rounded-full"></div>
    <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-indigo-500/10 blur-[120px] rounded-full"></div>
    <div className="absolute inset-0 bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] dark:bg-[radial-gradient(#ffffff0a_1px,transparent_1px)] opacity-50"></div>
  </div>
);
