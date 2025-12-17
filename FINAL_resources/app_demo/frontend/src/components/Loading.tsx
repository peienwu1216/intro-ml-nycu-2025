import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';

interface LoadingProps {
  isLoading: boolean;
}

const words = ["COMPOSITION", "LIGHTING", "BALANCE", "COLOR", "AESTHETICS"];

export default function Loading({ isLoading }: LoadingProps) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!isLoading) return;
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % words.length);
    }, 800);
    return () => clearInterval(interval);
  }, [isLoading]);

  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, transition: { duration: 0.8, ease: "easeInOut" } }}
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white"
        >
          <div className="relative flex flex-col items-center justify-center w-64 h-64">
            {/* Rotating geometric frame */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 border-[1px] border-neutral-200 rounded-full"
            />
            <motion.div
              animate={{ rotate: -360, scale: [1, 0.9, 1] }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
              className="absolute inset-4 border-[1px] border-neutral-300 rounded-full"
            />
            
            {/* Center Text */}
            <div className="z-10 flex flex-col items-center gap-4">
              <motion.div 
                className="h-[1px] w-12 bg-black"
                initial={{ width: 0 }}
                animate={{ width: 48 }}
                transition={{ duration: 1, repeat: Infinity, repeatType: "reverse" }}
              />
              
              <div className="h-6 overflow-hidden relative flex items-center justify-center w-40">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={words[index]}
                    initial={{ y: 20, opacity: 0, filter: "blur(5px)" }}
                    animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
                    exit={{ y: -20, opacity: 0, filter: "blur(5px)" }}
                    transition={{ duration: 0.5 }}
                    className="absolute font-serif text-sm tracking-[0.2em] text-black"
                  >
                    {words[index]}
                  </motion.span>
                </AnimatePresence>
              </div>

              <motion.div 
                className="h-[1px] w-12 bg-black"
                initial={{ width: 0 }}
                animate={{ width: 48 }}
                transition={{ duration: 1, repeat: Infinity, repeatType: "reverse", delay: 0.5 }}
              />
            </div>
          </div>
          
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.4 }}
            className="absolute bottom-12 font-sans text-xs tracking-widest text-neutral-500"
          >
            PROCESSING VISUAL DATA
          </motion.p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
