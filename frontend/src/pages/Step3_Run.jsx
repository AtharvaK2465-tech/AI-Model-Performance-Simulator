import { useState } from "react";
import { useStore } from "../store";
import { runSimulate } from "../api";

export default function Step3_Run({ onDone, onBack }) {
  const { dataset, selectedModels, distortionType, levels, setSimulationResults } = useStore();
  const [progress, setProgress] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    setProgress({ step: 0, total: levels.length });

    try {
      await runSimulate(
        { dataset, models: selectedModels, distortion_type: distortionType, levels },
        (p) => setProgress(p),
        (results) => {
          setSimulationResults(results);
          setRunning(false);
          setTimeout(onDone, 800);
        }
      );
    } catch (e) {
      setError("Simulation failed. Is the backend running on port 8000?");
      setRunning(false);
    }
  };

  const pct = progress ? Math.round((progress.step / progress.total) * 100) : 0;

  return (
    <div className="max-w-2xl mx-auto px-4">
      <h1 className="font-mono text-2xl text-[#00D4FF] mb-2">Run Simulation</h1>
      <p className="text-[#64748B] mb-8">Live progress streamed from the backend.</p>

      {/* Config summary */}
      <div className="bg-[#0F1629] border border-[#1E293B] rounded-xl p-5 mb-8 font-mono text-sm space-y-2">
        <div className="flex justify-between">
          <span className="text-[#64748B]">Dataset</span>
          <span className="text-white capitalize">{dataset.replace("_", " ")}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#64748B]">Models</span>
          <span className="text-white">{selectedModels.length} selected</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#64748B]">Distortion</span>
          <span className="text-[#00D4FF] capitalize">{distortionType}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#64748B]">Levels</span>
          <span className="text-white">{levels.join(" → ")}</span>
        </div>
      </div>

      {/* Progress bar */}
      {progress && (
        <div className="mb-8">
          <div className="flex justify-between font-mono text-sm mb-2">
            <span className="text-[#64748B]">
              {running
                ? `Level ${progress.step}/${progress.total} — distortion ${progress.level}`
                : "✓ Complete"}
            </span>
            <span className="text-[#00D4FF]">{pct}%</span>
          </div>
          <div className="w-full bg-[#1E293B] rounded-full h-2">
            <div
              className="bg-[#00D4FF] h-2 rounded-full transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          {!running && pct === 100 && (
            <p className="text-[#00D4FF] font-mono text-sm mt-3">
              ✓ Simulation complete — loading results...
            </p>
          )}
        </div>
      )}

      {error && (
        <p className="text-red-400 font-mono text-sm mb-6 bg-red-400/10 border border-red-400/20 rounded-lg p-3">
          {error}
        </p>
      )}

      <div className="flex gap-3">
        <button
          onClick={onBack}
          disabled={running}
          className="flex-1 py-3 rounded-xl border border-[#1E293B] text-[#64748B] font-mono hover:border-[#00D4FF]/40 transition-all disabled:opacity-40"
        >
          ← Back
        </button>
        <button
          onClick={handleRun}
          disabled={running}
          className="flex-[2] py-3 rounded-xl bg-[#00D4FF] text-[#0A0E1A] font-mono font-bold hover:bg-[#00D4FF]/90 transition-all disabled:opacity-40"
        >
          {running ? "Running..." : "▶ Start Simulation"}
        </button>
      </div>
    </div>
  );
}
