const STEPS = [
  { n: 1, label: "Dataset" },
  { n: 2, label: "Config" },
  { n: 3, label: "Run" },
  { n: 4, label: "Results" },
  { n: 5, label: "Analysis" },
  { n: 6, label: "Recommend" },
  { n: 7, label: "History" },
];

export default function StepNav({ current }) {
  return (
    <div className="flex items-center justify-center gap-2 py-6 px-4 flex-wrap">
      {STEPS.map((s, i) => {
        const done = s.n < current;
        const active = s.n === current;
        return (
          <div key={s.n} className="flex items-center gap-2">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-mono transition-all
              ${active ? "bg-[#00D4FF] text-[#0A0E1A] font-bold" : ""}
              ${done ? "bg-[#00D4FF]/20 text-[#00D4FF]" : ""}
              ${!active && !done ? "bg-[#0F1629] text-[#64748B] border border-[#1E293B]" : ""}
            `}>
              <span>{s.n}</span>
              <span className="hidden sm:inline">{s.label}</span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`w-5 h-px ${done ? "bg-[#00D4FF]" : "bg-[#1E293B]"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
