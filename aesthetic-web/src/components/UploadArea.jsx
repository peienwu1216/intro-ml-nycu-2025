import { forwardRef } from "react";
import { uploadAndGetScore } from "../api/scoreApi";

const UploadArea = forwardRef(({ onResult, onImageSelect }, ref) => {

  async function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return; 

    const imageUrl = URL.createObjectURL(file);

    onImageSelect(imageUrl); 

    try {
      const scoreData = await uploadAndGetScore(file);
      onResult(scoreData);
    } catch (err) {
      console.error('uploadAndGetScore failed:', err);
      // As a fallback, pass a simple error object so the UI can react.
      onResult({ success: false, error: true, message: err.message ?? 'Upload failed' });
    }
  }

  return (
    <input
      ref={ref}
      type="file"
      accept="image/*"
      onChange={handleFile}
      style={{ display: "none" }}
    />
  );

});

export default UploadArea;


  
