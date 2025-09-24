
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .retriever import FAISSLocalRetriever
from .rerank import CrossEncoderReranker, compress_with_reranker

# LangChain (optional)
HAS_LC = True
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableMap
    from langchain_openai import ChatOpenAI
except Exception:
    HAS_LC = False


@dataclass
class AppConfig:
    artifacts_dir: str
    embedding_model: str
    embedding_device: str
    retrieval_top_k: int
    retrieval_max_k: int
    generation_max_context_chars: int
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_config(cfg_path: str) -> AppConfig:
    with open(cfg_path, "r") as f:
        cfg = json.load(f) if cfg_path.endswith(".json") else _load_yaml_compat(f.read())

    if "artifacts_dir" not in cfg:
        raise KeyError("Missing 'artifacts_dir' in config.")
    embed_cfg = cfg.get("embedding", {})
    retrieval_cfg = cfg.get("retrieval", {})
    gen_cfg = cfg.get("generation", {})

    return AppConfig(
        artifacts_dir=cfg["artifacts_dir"],
        embedding_model=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device=embed_cfg.get("device", "cpu"),
        retrieval_top_k=int(retrieval_cfg.get("top_k", 5)),
        retrieval_max_k=int(retrieval_cfg.get("max_k", max(10, int(retrieval_cfg.get("top_k", 5))))),
        generation_max_context_chars=int(gen_cfg.get("max_context_chars", 6000)),
        rerank_model=cfg.get("rerank", {}).get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )


def _load_yaml_compat(text: str) -> Dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        data: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = _coerce(v.strip())
        return data


def _coerce(v: str) -> Any:
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except Exception:
        return v


class SimpleChain:
    """
    Retrieval -> (CrossEncoder reranker) -> Prompt -> LLM (optional)
    Gracefully falls back to extractive answer (top reranked chunk) when LLM unavailable or errors (429, etc).
    """

    def __init__(
        self,
        retriever: FAISSLocalRetriever,
        reranker: CrossEncoderReranker,
        cfg: AppConfig,
        use_lcel: bool = True,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.cfg = cfg

        # use_lcel is allowed only if LC and OPENAI are available AND user didn't force disable
        self.has_openai = bool(os.environ.get("OPENAI_API_KEY")) and HAS_LC
        self.use_lcel = bool(use_lcel and self.has_openai)

        self.prompt = None
        self.llm = None
        if self.use_lcel:
            self._init_lcel()

    def _init_lcel(self):
        self.prompt = ChatPromptTemplate.from_template(
            "Answer concisely and ONLY using the provided context. If unsure, say you don't know.\n\n"
            "Context:\n{context}\n\nQ: {question}\n"
        )
        self.llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.0")),
        )

        def _retrieve(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
            q = inputs["question"]
            company = inputs.get("company")
            year = inputs.get("year")
            k = int(inputs.get("k") or self.cfg.retrieval_max_k)
            docs = self.retriever.search(q, company=company, year=year, top_k=k)
            return docs

        def _compress(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
            q = inputs["question"]
            docs = inputs["docs"]
            top_k = int(inputs.get("top_k") or self.cfg.retrieval_top_k)
            return compress_with_reranker(q, docs, self.reranker, k=top_k)

        def _format_ctx(inputs: Dict[str, Any]) -> str:
            docs = inputs["compressed_docs"]
            text = "\n\n".join(d["text"] for d in docs)
            if len(text) > self.cfg.generation_max_context_chars:
                text = text[: self.cfg.generation_max_context_chars]
            return text

        self.retrieve_runnable = RunnableLambda(_retrieve)
        self.compress_runnable = RunnableLambda(_compress)
        self.format_ctx_runnable = RunnableLambda(_format_ctx)

        self.pipeline = (
            RunnableMap({
                "question": RunnableLambda(lambda x: x["question"]),
                "company": RunnableLambda(lambda x: x.get("company")),
                "year": RunnableLambda(lambda x: x.get("year")),
                "k": RunnableLambda(lambda x: x.get("k")),
                "top_k": RunnableLambda(lambda x: x.get("top_k")),
            })
            | RunnableLambda(lambda x: {"question": x["question"], "docs": self.retrieve_runnable.invoke(x)})
            | RunnableLambda(lambda x: {**x, "compressed_docs": self.compress_runnable.invoke({"question": x["question"], "docs": x["docs"], "top_k": x.get("top_k")})})
            | RunnableLambda(lambda x: {**x, "context": self.format_ctx_runnable.invoke({"compressed_docs": x["compressed_docs"]})})
            | RunnableLambda(lambda x: {"prompt": self.prompt.format(context=x["context"], question=x["question"]), **x})
        )

    def _extractive_answer(self, question: str, company: Optional[str], year: Optional[int], top_k: int) -> Dict[str, Any]:
        docs = self.retriever.search(question, company=company, year=year, top_k=self.cfg.retrieval_max_k)
        top_docs = compress_with_reranker(question, docs, self.reranker, k=top_k or self.cfg.retrieval_top_k)
        answer = top_docs[0]["text"] if top_docs else ""
        contexts = [d["text"] for d in top_docs]
        return {"answer": answer, "contexts": contexts, "meta": {"company": company, "year": year}}

    def invoke(self, question: str, company: Optional[str], year: Optional[int], top_k: int) -> Dict[str, Any]:
        # If LCEL disabled or no OpenAI, do extractive path
        if not self.use_lcel or self.llm is None or self.prompt is None:
            return self._extractive_answer(question, company, year, top_k)

        # LCEL path with runtime fallback
        try:
            inputs = {
                "question": question,
                "company": company,
                "year": year,
                "k": self.cfg.retrieval_max_k,
                "top_k": top_k or self.cfg.retrieval_top_k,
            }
            stage = self.pipeline.invoke(inputs)
            prompt_val = stage["prompt"]
            compressed_docs = stage["compressed_docs"]
            resp = self.llm.invoke(prompt_val)  # may raise due to 429/quotas
            answer = getattr(resp, "content", str(resp))
            contexts = [d["text"] for d in compressed_docs]
            return {"answer": answer, "contexts": contexts, "meta": {"company": company, "year": year}}
        except Exception as e:
            # Fallback gracefully on any LLM error (429, timeouts, etc.)
            # Optional: log to stderr for visibility
            print(f"[warn] LLM failed ({type(e).__name__}): {e}. Falling back to extractive answer.", flush=True)
            self.use_lcel = False
            return self._extractive_answer(question, company, year, top_k)

    def run(self, question: str, company: Optional[str] = None, year: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
        return self.invoke(question, company, year, top_k)


def load_chain(cfg_path: str, use_lcel: bool = True) -> SimpleChain:
    cfg = load_config(cfg_path)
    retriever = FAISSLocalRetriever(
        faiss_index_path=os.path.join(cfg.artifacts_dir, "sec_faiss.index"),
        chunks_meta_path=os.path.join(cfg.artifacts_dir, "chunks.pkl"),
        embed_model=cfg.embedding_model,
        embed_device=cfg.embedding_device,
    )
    reranker = CrossEncoderReranker(model_name=cfg.rerank_model, top_k=cfg.retrieval_top_k)
    return SimpleChain(retriever=retriever, reranker=reranker, cfg=cfg, use_lcel=use_lcel)
