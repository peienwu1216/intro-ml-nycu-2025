import { motion, AnimatePresence } from 'framer-motion';
import { useMemo, useState, useEffect, useRef } from 'react';
import { clsx } from 'clsx';
import { ScanEye } from 'lucide-react';
import type { AnalysisResult } from '../types';

interface ResultsPageProps {
  files: File[];
  results: AnalysisResult[];
}

const GradCamView = ({ src, gradcamSrc, isActive, onToggle }: { src: string, gradcamSrc: string | null, isActive: boolean, onToggle: () => void }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isHinting, setIsHinting] = useState(false);
  const [suppressHint, setSuppressHint] = useState(false);
  const hasHinted = useRef(false);

  // Auto-hint animation: Only once
  useEffect(() => {
    if (isActive || hasHinted.current) return;
    
    const timer = setTimeout(() => {
      if (isActive || hasHinted.current) return;
      setIsHinting(true);
      
      setTimeout(() => {
        setIsHinting(false);
        hasHinted.current = true;
      }, 1500);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [isActive]);

  const isPeeking = (isHovered && !suppressHint) || isHinting;

  const handleToggle = () => {
    if (isActive) {
      setSuppressHint(true);
    }
    onToggle();
  };

  return (
    <div 
      className="relative w-full h-full flex items-center justify-center cursor-pointer group"
      onClick={handleToggle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setSuppressHint(false);
      }}
    >
      {/* Bottom Layer: Grad-CAM */}
      <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
        {gradcamSrc ? (
            <img 
            src={gradcamSrc} 
            alt="GradCAM" 
            className="max-w-full max-h-full object-contain"
            />
        ) : (
            <div className="text-neutral-400 text-xs">Grad-CAM not available</div>
        )}
        
        <div className="absolute top-4 left-4 bg-black/80 text-white text-[10px] px-2 py-1 font-sans tracking-widest">
          GRAD-CAM ANALYSIS
        </div>
      </div>

      {/* Top Layer: Original Image (The Cover) */}
      <motion.div 
        className="absolute inset-0 flex items-center justify-center bg-neutral-50 overflow-hidden"
        animate={{ 
          y: isActive ? '-100%' : (isPeeking ? -40 : 0),
          rotateX: isActive ? 20 : (isPeeking ? 5 : 0),
          scale: isActive ? 0.9 : 1
        }}
        transition={{ 
          type: "spring", 
          stiffness: 100, 
          damping: 20,
          mass: 1
        }}
        style={{ transformOrigin: "top center" }}
      >
        <img 
          src={src} 
          alt="Original" 
          className="max-w-full max-h-full object-contain shadow-xl"
        />
        
        {/* Hint Label */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: isPeeking && !isActive ? 1 : 0 }}
          className="absolute bottom-8 bg-white/90 backdrop-blur px-4 py-2 rounded-full flex items-center gap-2 shadow-lg"
        >
          <ScanEye className="w-4 h-4" />
          <span className="font-sans text-[10px] tracking-widest font-bold">VIEW ATTENTION MAP</span>
        </motion.div>
      </motion.div>
    </div>
  );
};

const WaveChart = ({ distribution }: { distribution: number[] }) => {
  const points = distribution.map((val, i) => ({ x: i, y: val }));
  
  // Create a smooth path
  const pathD = useMemo(() => {
    const width = 100;
    const height = 50;
    const step = width / (points.length - 1);
    
    let d = `M 0 ${height}`; // Start bottom left
    
    // Curve points
    points.forEach((p, i) => {
      const x = i * step;
      const y = height - (p.y * height); // Invert Y for SVG
      
      if (i === 0) {
        d += ` L ${x} ${y}`;
      } else {
        // Simple cubic bezier for smoothness
        const prevX = (i - 1) * step;
        const prevY = height - (points[i-1].y * height);
        const cp1x = prevX + step / 2;
        const cp1y = prevY;
        const cp2x = x - step / 2;
        const cp2y = y;
        d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x} ${y}`;
      }
    });
    
    d += ` L ${width} ${height} Z`; // Close path
    return d;
  }, [points]);

  return (
    <div className="w-full h-24 relative mt-4">
      <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full overflow-visible">
        {/* Gradient Definition */}
        <defs>
          <linearGradient id="waveGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="black" stopOpacity="0.2" />
            <stop offset="100%" stopColor="black" stopOpacity="0" />
          </linearGradient>
        </defs>
        
        {/* Area Path */}
        <motion.path
          d={pathD}
          fill="url(#waveGradient)"
          initial={{ d: "M 0 50 L 100 50 Z" }}
          animate={{ d: pathD }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
        
        {/* Line Path */}
        <motion.path
          d={pathD.replace(' Z', '').replace(`L 100 50`, '')} // Remove closure for line
          fill="none"
          stroke="black"
          strokeWidth="0.5"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
        />
      </svg>
    </div>
  );
};

export default function ResultsPage({ files, results }: ResultsPageProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isGradCamActive, setIsGradCamActive] = useState(false);
  
  const result = results[selectedIndex];

  // Reset GradCam state when switching images
  useEffect(() => {
    setIsGradCamActive(false);
  }, [selectedIndex]);

  if (!result) return null;

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full h-full flex flex-col lg:flex-row overflow-hidden bg-white"
    >
      {/* Left: Image Container */}
      <div className="flex-shrink-0 lg:flex-1 relative bg-neutral-50 flex flex-col items-center justify-center p-4 lg:p-12 overflow-hidden h-[40vh] lg:h-auto border-b lg:border-b-0 lg:border-r border-neutral-100 perspective-[1000px]">
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedIndex}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ duration: 0.4 }}
            className="w-full h-full flex items-center justify-center"
          >
            <GradCamView 
              src={URL.createObjectURL(files[selectedIndex])} 
              gradcamSrc={result.gradcam_image}
              isActive={isGradCamActive}
              onToggle={() => setIsGradCamActive(!isGradCamActive)}
            />
          </motion.div>
        </AnimatePresence>

        {/* Thumbnails Overlay (Bottom Center) */}
        {files.length > 1 && (
          <div className="absolute bottom-4 lg:bottom-8 left-1/2 -translate-x-1/2 flex gap-2 p-2 bg-white/80 backdrop-blur-md rounded-full shadow-sm z-20 max-w-[90%] overflow-x-auto">
            {files.map((file, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIndex(idx)}
                className={clsx(
                  "w-8 h-8 lg:w-12 lg:h-12 flex-shrink-0 rounded-full overflow-hidden border-2 transition-all duration-300",
                  selectedIndex === idx ? "border-black scale-110" : "border-transparent opacity-60 hover:opacity-100"
                )}
              >
                <img 
                  src={URL.createObjectURL(file)} 
                  alt={`Thumbnail ${idx}`} 
                  className="w-full h-full object-cover"
                />
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Right: Analysis Panel */}
      <div className="flex-shrink-0 lg:w-[400px] xl:w-[500px] h-full overflow-y-auto bg-white p-8 lg:p-12 flex flex-col">
        <motion.div
          key={`info-${selectedIndex}`}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="font-serif text-4xl mb-2">Aesthetic Score</h2>
          <div className="flex items-baseline gap-4 mb-8">
            <span className="text-6xl font-bold tracking-tighter">{result.score}</span>
            <span className="text-neutral-400 font-sans text-sm uppercase tracking-widest">/ 5.0000</span>
          </div>

          <div className="mb-12">
            <h3 className="font-sans text-xs font-bold uppercase tracking-widest mb-4 text-neutral-900">Score Distribution</h3>
            <WaveChart distribution={result.distribution} />
            <div className="flex justify-between mt-2 text-[10px] text-neutral-400 font-sans">
              <span>1.0</span>
              <span>2.0</span>
              <span>3.0</span>
              <span>4.0</span>
              <span>5.0</span>
            </div>
          </div>

          <div>
            <h3 className="font-sans text-xs font-bold uppercase tracking-widest mb-6 text-neutral-900">Composition Attributes</h3>
            <div className="space-y-6">
              {result.attributes.map((attr, i) => (
                <div key={attr.name} className="group">
                  <div className="flex justify-between mb-2">
                    <span className="font-serif text-lg text-neutral-600 group-hover:text-black transition-colors">{attr.name}</span>
                    <span className="font-mono text-sm text-neutral-400">{attr.value.toFixed(4)}</span>
                  </div>
                  <div className="h-[1px] w-full bg-neutral-100 overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${attr.value * 100}%` }}
                      transition={{ duration: 1, delay: 0.4 + (i * 0.1) }}
                      className="h-full bg-black"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
