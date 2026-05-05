const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function runSimulate(payload, onProgress, onDone) {
  const response = await fetch(`${BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split("\n").filter(l => l.startsWith("data: "));

    for (const line of lines) {
      const data = JSON.parse(line.replace("data: ", ""));
      if (data.done) onDone(data.results);
      else if (!data.error) onProgress(data);
    }
  }
}

export async function runAnalysis(results) {
  const res = await fetch(`${BASE}/api/analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ results }),
  });
  return res.json();
}

export async function runRecommend(results, weights) {
  const res = await fetch(`${BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ results, weights }),
  });
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${BASE}/api/history`);
  return res.json();
}
