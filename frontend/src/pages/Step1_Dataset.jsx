import { useStore } from "../store";

const DATASETS = [
  { id: "iris", name: "Iris", desc: "150 samples · 4 features · 3 classes", icon: "🌸" },
  { id: "breast_cancer", name: "Breast Cancer", desc: "569 samples · 30 features · 2 classes", icon: "🔬" },
];

export default function Step1_Dataset({ onNext }) {
  const { dataset, setDataset } = useStore();

  return (
    <div className="max-w-2xl mx-auto px-4">
      <h1 className="font-mono text-2xl text-[#00D4FF] mb-2">Select Dataset</h1>
      <p className="text-[#64748B] mb-8">Choose the dataset to run your simulation on.</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-10">
        {DATASETS.map((d) => (
          <button
            key={d.id}
            onClick={() => setDataset(d.id)}
            className={`text-left p-5 rounded-xl border transition-all
              ${dataset === d.id
                ? "border-[#00D4FF] bg-[#00D4FF]/10 shadow-lg"
                : "border-[#1E293B] bg-[#0F1629] hover:border-[#00D4FF]/40"
              }`}
          >
            <div className="text-3xl mb-3">{d.icon}</div>
            <div className="font-mono text-white font-bold mb-1">{d.name}</div>
            <div className="text-[#64748B] text-sm">{d.desc}</div>
          </button>
        ))}
      </div>

      <button
        onClick={onNext}
        className="w-full py-3 rounded-xl bg-[#00D4FF] text-[#0A0E1A] font-mono font-bold hover:bg-[#00D4FF]/90 transition-all"
      >
        Continue →
      </button>
    </div>
  );
}
