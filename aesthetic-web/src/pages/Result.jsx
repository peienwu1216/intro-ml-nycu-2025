import RadarScore from "../components/RadarScore.jsx";

export default function ResultPage({ result, previewUrl, onReset }) {
  return (
    <div className="result-container">
        
        <div className="result-layout">
            
            <div className="image-box">
                <img 
                    src={previewUrl} 
                    alt="Uploaded" 
                    className="uploaded-image" 
                />
            </div>

            <div className="score-box">
                <h2 className="text-3xl font-bold mb-4">Aesthetic Score</h2>
                
                <RadarScore data={result} />
                
                <button 
                    onClick={onReset}
                    className="mt-8 px-6 py-3 rounded-xl font-bold bg-gray-200 hover:bg-gray-300 transition"
                >
                    Try Another Image
                </button>
            </div>
        </div>
    </div>
  );
}