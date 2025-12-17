import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Header from './components/Header';
import Hero from './components/Hero';
import UploadSection from './components/UploadSection';
import Loading from './components/Loading';
import ResultsPage from './components/ResultsPage';
import BackgroundGallery from './components/BackgroundGallery';
import type { AnalysisResult } from './types';

type ViewState = 'home' | 'analyzing' | 'results';

function App() {
  const [view, setView] = useState<ViewState>('home');
  const [analyzedFiles, setAnalyzedFiles] = useState<File[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);

  const handleAnalyze = async (files: File[]) => {
    setView('analyzing');
    
    const newResults: AnalysisResult[] = [];
    const successfulFiles: File[] = [];
    const errors: string[] = [];

    for (const file of files) {
      // Client-side validation
      if (file.size > 10 * 1024 * 1024) {
        errors.push(`File ${file.name} is too large (max 10MB)`);
        continue;
      }

      const formData = new FormData();
      formData.append('file', file);

      try {
        // Use relative path for production (served by same origin)
        // Or environment variable for dev
        const apiUrl = import.meta.env.PROD ? '/analyze' : 'http://localhost:5001/analyze';
        
        const response = await fetch(apiUrl, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          const errorMessage = errorData.error || `Error ${response.status}`;
          console.error(`Error analyzing ${file.name}:`, errorMessage);
          errors.push(`${file.name}: ${errorMessage}`);
          continue;
        }

        const data = await response.json();
        newResults.push(data);
        successfulFiles.push(file);
      } catch (error) {
        console.error(`Network error analyzing ${file.name}:`, error);
        errors.push(`${file.name}: Network error`);
      }
    }
    
    if (errors.length > 0) {
      alert(`Some files could not be processed:\n${errors.join('\n')}`);
    }

    if (newResults.length > 0) {
      setAnalyzedFiles(successfulFiles);
      setResults(newResults);
      setView('results');
    } else {
      setAnalyzedFiles([]);
      setResults([]);
      setView('home'); // Go back if all failed
    }
  };

  const handleBack = () => {
    setAnalyzedFiles([]);
    setResults([]);
    setView('home');
  };

  return (
    <div className="h-screen w-screen bg-white text-black selection:bg-black selection:text-white overflow-hidden flex flex-col relative">
      <Loading isLoading={view === 'analyzing'} />
      
      <Header onHomeClick={handleBack} />
      
      <main className="flex-1 relative w-full h-full flex flex-col">
        <AnimatePresence mode="wait">
          {view === 'home' && (
            <motion.div 
              key="home"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0.5 } }}
              className="relative w-full h-full flex flex-col lg:flex-row items-center justify-center px-6 lg:px-24 gap-8 lg:gap-12 overflow-y-auto lg:overflow-hidden py-24 lg:py-0"
            >
              {/* Background Gallery only on Home */}
              <BackgroundGallery />

              <div className="w-full max-w-7xl mx-auto flex flex-col lg:flex-row items-center justify-center gap-12 lg:gap-24">
                <div className="relative z-10 flex-1 flex justify-center lg:justify-end w-full max-w-xl lg:max-w-none">
                  <Hero />
                </div>
                <div className="relative z-10 flex-1 w-full max-w-xl flex items-center justify-center lg:justify-start">
                  <UploadSection onAnalyze={handleAnalyze} />
                </div>
              </div>
            </motion.div>
          )}

          {view === 'results' && (
            <motion.div
              key="results"
              className="w-full h-full pt-20" // pt-20 to account for fixed header
            >
              <ResultsPage files={analyzedFiles} results={results} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer only on Home view */}
      {view === 'home' && (
        <motion.footer 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute bottom-0 left-0 right-0 py-6 px-8 border-t border-neutral-100/50 bg-white/60 backdrop-blur-sm z-20"
        >
          <div className="flex justify-between items-center">
            <span className="font-sans text-[10px] text-neutral-500 tracking-widest">
              © 2025 MACHINE LEARNING PROJECT
            </span>
          </div>
        </motion.footer>
      )}
    </div>
  );
}

export default App;
