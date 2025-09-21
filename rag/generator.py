from functools import lru_cache
from transformers import pipeline

@lru_cache(maxsize=1)
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

PROMPT = (
    "Answer strictly from the context. Use one short noun phrase (3–8 words). "
    "If uncertain, say 'not found in context'.\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)

def _pack_context(chunks, tok, limit_tokens=480):
    acc_ids, parts = [], []
    for ch in chunks:
        ids = tok.encode(ch, add_special_tokens=False)
        remain = limit_tokens - len(acc_ids)
        if remain <= 0:
            break
        take = ids[:remain]
        acc_ids.extend(take)
        parts.append(tok.decode(take, skip_special_tokens=True))
        if len(acc_ids) >= limit_tokens:
            break
    return "\n\n".join(parts)

def generate_answer(context_chunks, question, max_new_tokens=24):
    gen = get_generator()
    tok = gen.tokenizer
    ctx = _pack_context(context_chunks, tok, limit_tokens=480)
    text = PROMPT.format(ctx=ctx, q=question)
    out = gen(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )[0]["generated_text"]
    return out.split("Answer:")[-1].strip()
