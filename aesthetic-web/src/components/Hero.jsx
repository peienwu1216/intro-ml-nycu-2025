export default function Hero({ onUploadClick }) {
  return (
    <div className="container">
        <div className="box2">
           <h1 className="text-4xl font-bold" style={{ marginBottom: "0px" }}>
             Free online aesthetic image rating.
           </h1>
           <p className="text-gray-700 text-lg barlow-regular">
             Upload your image and get an instant aesthetic score. Our intelligent algorithm evaluates Composition, Lighting, Clarity, and Story.
           </p>
           <button 
             onClick={onUploadClick}
             className="px-8 py-4 rounded-2xl inline-block text-lg font-bold cursor-pointer hover:opacity-90 transition"
           >
             Upload Your Image
           </button>
        </div>
        
        <div className="flex box1">
         <img
          src="src/assets/sample_photo.jpeg"
          alt="hero"
        />
      </div>
        
    </div>
    
  );
}
