import { useState, useRef } from "react";
import Hero from "./components/Hero";
import UploadArea from "./components/UploadArea";
import ResultPage from "./pages/Result";
import RadarScore from "./components/RadarScore";

export default function App() {
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const fileInputRef = useRef(null);

  const handleTriggerUpload = () => {
    fileInputRef.current?.click();
  };

  const handleReset = () => {
    setResult(null);
    setPreviewUrl(null);
  };
  return (
    <div> 
      
      <UploadArea 
        ref={fileInputRef} 
        onResult={setResult} 
        onImageSelect={setPreviewUrl} 
      />
      
      {!result ? (
        <Hero onUploadClick={handleTriggerUpload} />
      ) : (
        <ResultPage 
            result={result} 
            previewUrl={previewUrl} 
            onReset={handleReset} 
        />
      )}

    </div>
  );
}


