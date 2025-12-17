export interface AnalysisResult {
  score: number;
  distribution: number[];
  attributes: { name: string; value: number }[];
  gradcam_image: string | null;
}
