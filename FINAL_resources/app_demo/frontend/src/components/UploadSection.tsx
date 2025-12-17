import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, ArrowRight, Image as ImageIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface UploadSectionProps {
  onAnalyze: (files: File[]) => void;
}

export default function UploadSection({ onAnalyze }: UploadSectionProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      handleFiles(selectedFiles);
    }
  };

  const handleFiles = (newFiles: File[]) => {
    const validFiles = newFiles.filter(file => file.type.startsWith('image/'));
    // Since we auto-analyze immediately, we should treat each upload as a new batch
    // instead of accumulating them, which causes issues if previous files had errors.
    const currentBatch = validFiles.slice(0, 5);
    setFiles(currentBatch);
    
    // Auto-analyze immediately when files are added
    if (currentBatch.length > 0) {
      onAnalyze(currentBatch);
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  return (
    <section className="w-full h-full flex flex-col justify-center">
      <AnimatePresence mode="wait">
        <motion.div
            key="upload-zone"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
            className={clsx(
              "relative w-full h-[300px] md:h-[400px] border-[1px] border-dashed transition-all duration-500 flex flex-col items-center justify-center cursor-pointer group overflow-hidden",
              isDragging ? "border-black bg-neutral-50" : "border-neutral-300 hover:border-neutral-400"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            {/* Removed background image on hover */}
            
            <div className="z-10 flex flex-col items-center gap-6">
              <div className="w-16 h-16 rounded-full border-[1px] border-neutral-300 flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                <Upload className="w-6 h-6 text-neutral-400 group-hover:text-black transition-colors" />
              </div>
              <div className="text-center">
                <h3 className="font-serif text-2xl text-black mb-2">Upload Your Vision</h3>
                <p className="font-sans text-xs tracking-widest text-neutral-400 uppercase">
                  Drag & Drop or Click to Browse
                </p>
                <p className="font-sans text-[10px] text-neutral-300 mt-2">
                  Max 5 Images • JPG, PNG, WEBP, HEIC
                </p>
              </div>
            </div>
          </motion.div>
      </AnimatePresence>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInput}
        className="hidden"
        multiple
        accept="image/*"
      />
    </section>
  );
}
