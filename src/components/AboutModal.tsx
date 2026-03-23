import React, { useEffect, useState } from 'react';
import { getVersion } from '@tauri-apps/api/app';
import { logger } from '../lib/logger';

const log = logger.scope('App');

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AboutModal: React.FC<AboutModalProps> = ({ isOpen, onClose }) => {
  const [appVersion, setAppVersion] = useState<string>('0.4.0'); // Fallback/Dev version

  useEffect(() => {
    getVersion().then(setAppVersion).catch((err) => log.error('Failed to get app version', { error: err }));
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div 
        className="bg-[#1a1a1a] border border-[#333] rounded-lg p-8 max-w-md w-full shadow-2xl relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 text-[#666] hover:text-white transition-colors"
        >
          ✕
        </button>

        <div className="flex flex-col items-center text-center">
          <img src="/crebain.png" alt="Crebain Logo" className="w-24 h-24 mb-6" />
          
          <h2 className="text-2xl font-bold text-white tracking-wider mb-2">CREBAIN</h2>
          <p className="text-[#888] text-sm tracking-widest uppercase mb-6">Adaptive Response & Awareness System</p>
          
          <div className="space-y-2 text-[#ccc] text-sm mb-8">
            <p>Version {appVersion}</p>
            <p>System Core v4.0.0</p>
            <p className="pt-2 text-[#666]">
              Professional-grade tactical reconnaissance platform with 3D Gaussian Splatting visualization and advanced multi-modal sensor fusion.
            </p>
          </div>

          <div className="text-xs text-[#444]">
            <p>© 2026 Gitjo. All rights reserved.</p>
            <p className="mt-1">Built with Tauri 2 & React 19</p>
          </div>
        </div>
      </div>
    </div>
  );
};
