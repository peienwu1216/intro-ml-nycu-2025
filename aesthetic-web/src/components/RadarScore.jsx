import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts";

export default function RadarScore({ data }) {
  const radarData = [
     { subject: "Composition", value: 86},
    { subject: "Lighting", value: 33 },
    { subject: "Clarity", value: 51 },
    { subject: "Story", value: 78 },
    
    // { subject: "Composition", value: data.composition },
    // { subject: "Lighting", value: data.lighting },
    // { subject: "Clarity", value: data.clarity },
    // { subject: "Story", value: data.story },
  ];

  return (
    <div className="flex flex-col items-center mt-10">
      <h2 className="text-3xl font-bold">Aesthetic Score: {(data.overall * 100).toFixed(1)}</h2>

      <RadarChart cx={200} cy={200} outerRadius={150} width={400} height={400} data={radarData}>
        <PolarGrid />
        <PolarAngleAxis dataKey="subject" />
        <PolarRadiusAxis angle={30} domain={[0, 1]} />
        <Radar name="Score" dataKey="value" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.6} />
      </RadarChart>
    </div>
  );
}
