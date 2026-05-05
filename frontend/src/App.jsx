import { useState } from "react";
import StepNav from "./components/StepNav";
import Step1_Dataset from "./pages/Step1_Dataset";
import Step2_Config from "./pages/Step2_Config";
import Step3_Run from "./pages/Step3_Run";

export default function App() {
  const [step, setStep] = useState(1);

  return (
    <div className="min-h-screen bg-[#0A0E1A] text-white">
      <div className="max-w-4xl mx-auto">
        <div className="text-center pt-10 pb-2">
          <h1 className="font-mono text-3xl font-bold text-[#00D4FF] tracking-tight">
            AI Model Performance Simulator
          </h1>
          <p className="text-[#64748B] mt-2 text-sm">
            Test how your models hold up under real-world data degradation
          </p>
        </div>

        <StepNav current={step} />

        <div className="py-6">
          {step === 1 && <Step1_Dataset onNext={() => setStep(2)} />}
          {step === 2 && <Step2_Config onNext={() => setStep(3)} onBack={() => setStep(1)} />}
          {step === 3 && <Step3_Run onDone={() => setStep(4)} onBack={() => setStep(2)} />}
          {step >= 4 && (
            <div className="text-center font-mono text-[#00D4FF] py-20">
              ✓ Results ready — Step 4 coming in Session 3
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
