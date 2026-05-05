import { useStore } from "../store";

const ALL_MODELS = [
  "Random Forest", "Logistic Regression", "SVM",
  "Decision Tree", "KNN", "Gradient Boosting", "XGBoost"
];

const DISTORTIONS = [
  { id: "noise", label: "Gaussian Noise" },
  { id: "drift", label: "Data Drift" },
  { id: "missing", label: "Missing Values" },
  { id: "imbalance", label: "Class Imbalance" },
  { id: "outlier", label: "Outliers" },
];

export default function Step2_Config({ onNext, onBack }) {
  const {
    selectedModels, setSelectedModels,
    distortionType, setDistortionType,
    levels, setLevels,
  } = useStore();

  const toggleModel = (m) => {
    setSelectedModels(
      selectedModels.includes(m)
        ? selectedModels.filter(x => x !== m)
        : [...selectedModels, m]
    );
  };

  const handleLevels = (e) => {
    const max = parseFloat(e.target.value);
    const arr = Array.from({ length: 5 }, (_, i) =>
      parseFloat((i * max / 4).toFixed(2))
    );
    setLevels(arr);
  };

  return (
    <div className="max-w-2xl mx-auto px-4">
      <h1 className="font-mono text-2xl text-[#00D4FF] mb-2">Configure Simulation</h1>
      <p className="text-[#64748B] mb-8">Select models and distortion settings.</p>

      {/* Models */}
      <div className="mb-8">
        <h2 className="font-mono text-white mb-3">Models</h2>
        <div className="flex flex-wrap gap-2">
          {ALL_MODELS.map((m) => (
            <button
              key={m}
              onClick={() => toggleModel(m)}
              className={`px-3 py-1.5 rounded-lg text-sm font-mono border transition-all
                ${selectedModels.includes(m)
                  ? "border-[#00D4FF] bg-[#00D4FF]/10 text-[#00D4FF]"
                  : "border-[#1E293B] bg-[#0F1629] text-[#64748B] hover:border-[#00D4FF]/40"
                }`}
            >
              {m}
            </button>
          ))}
        </div>
        {selectedModels.length === 0 && (
          <p className="text-red-400 text-sm mt-2">Select at least one model</p>
        )}
      </div>

      {/* Distortion type */}
      <div className="mb-8">
        <h2 className="font-mono text-white mb-3">Distortion Type</h2>
        <div className="flex flex-wrap gap-2">
          {DISTORTIONS.map((d) => (
            <button
              key={d.id}
              onClick={() => setDistortionType(d.id)}
              className={`px-3 py-1.5 rounded-lg text-sm font-mono border transition-all
                ${distortionType === d.id
                  ? "border-[#00D4FF] bg-[#00D4FF]/10 text-[#00D4FF]"
                  : "border-[#1E293B] bg-[#0F1629] text-[#64748B] hover:border-[#00D4FF]/40"
                }`}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      {/* Max distortion level */}
      <div className="mb-10">
        <h2 className="font-mono text-white mb-3">
          Max Distortion Level —{" "}
          <span className="text-[#00D4FF]">{levels[levels.length - 1]}</span>
        </h2>
        <input
          type="range" min="0.1" max="0.8" step="0.1"
          defaultValue={levels[levels.length - 1]}
          onChange={handleLevels}
          className="w-full accent-[#00D4FF]"
        />
        <div className="flex justify-between text-[#64748B] text-xs mt-2 font-mono">
          {levels.map(l => <span key={l}>{l}</span>)}
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={onBack}
          className="flex-1 py-3 rounded-xl border border-[#1E293B] text-[#64748B] font-mono hover:border-[#00D4FF]/40 transition-all"
        >
          ← Back
        </button>
        <button
          onClick={onNext}
          disabled={selectedModels.length === 0}
          className="flex-[2] py-3 rounded-xl bg-[#00D4FF] text-[#0A0E1A] font-mono font-bold hover:bg-[#00D4FF]/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Run Simulation →
        </button>
      </div>
    </div>
  );
}
