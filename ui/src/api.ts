export async function queryApi(q: {text:string; company:string; year:number}) {
  const r = await fetch(`${import.meta.env.VITE_API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": import.meta.env.VITE_API_KEY!,
    },
    body: JSON.stringify({...q, no_openai: true, top_k: 3}),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
