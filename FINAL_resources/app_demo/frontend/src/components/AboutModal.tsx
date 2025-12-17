import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AboutModal({ isOpen, onClose }: AboutModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-[60]"
          />
          <motion.div
            initial={{ opacity: 0, y: 100, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 100, scale: 0.95 }}
            className="fixed inset-0 m-auto w-full max-w-2xl h-[80vh] bg-white shadow-2xl z-[70] overflow-hidden flex flex-col"
          >
            <div className="flex justify-between items-center p-6 border-b border-neutral-100">
              <h2 className="font-serif text-2xl">About the Project</h2>
              <button onClick={onClose} className="p-2 hover:bg-neutral-100 rounded-full transition-colors">
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-8 font-sans leading-relaxed text-neutral-600 space-y-6">
              <section>
                <h3 className="font-bold text-black mb-2 uppercase tracking-widest text-xs">The Vision</h3>
                <p>
                  Unlike traditional machine learning tasks that seek objective truth (e.g., "Is this a cat?"), 
                  Aesthetic Assessment models subjective perception. We aim to model <strong>Human Consensus</strong> on beauty, 
                  aggregating opinions rather than finding a single "correct" answer.
                </p>
              </section>

              <section>
                <h3 className="font-bold text-black mb-2 uppercase tracking-widest text-xs">The Data (CADB)</h3>
                <p>
                  We utilize the <strong>Composition-aware Aesthetic Database (CADB)</strong>, containing 9,497 images. 
                  Each image is rated by 5 experts, providing a score distribution that captures both consensus and controversy. 
                  The dataset also includes 12 specific photographic attributes like Rule of Thirds, Symmetry, and Depth of Field.
                </p>
              </section>

              <section>
                <h3 className="font-bold text-black mb-2 uppercase tracking-widest text-xs">The Model</h3>
                <p>
                  Our architecture, <strong>ConvNeXt V2 Nano</strong>, combines the efficiency of CNNs with Transformer design philosophies. 
                  Key innovations include:
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1 ml-2">
                  <li><strong>SGFM (Saliency-Guided Feature Modulation):</strong> Uses saliency maps to focus on important visual elements.</li>
                  <li><strong>Rank Loss:</strong> Optimizes for relative ranking order rather than just absolute score accuracy.</li>
                  <li><strong>Label Distribution Learning:</strong> Predicts the probability of each score (1-5) to model aesthetic ambiguity.</li>
                </ul>
              </section>

              <section>
                <h3 className="font-bold text-black mb-2 uppercase tracking-widest text-xs">Performance</h3>
                <p>
                  Our model achieves a Spearman Rank Correlation Coefficient (SRCC) of <strong>0.65</strong>, 
                  outperforming previous state-of-the-art models like SAMPNet.
                </p>
              </section>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
