import { create } from "zustand";

export const useStore = create((set) => ({
  dataset: "iris",
  setDataset: (v) => set({ dataset: v }),

  selectedModels: ["Random Forest", "Logistic Regression"],
  setSelectedModels: (v) => set({ selectedModels: v }),

  distortionType: "noise",
  setDistortionType: (v) => set({ distortionType: v }),

  levels: [0.0, 0.1, 0.2, 0.3, 0.4],
  setLevels: (v) => set({ levels: v }),

  simulationResults: null,
  setSimulationResults: (v) => set({ simulationResults: v }),

  currentStep: 1,
  setCurrentStep: (v) => set({ currentStep: v }),
}));
