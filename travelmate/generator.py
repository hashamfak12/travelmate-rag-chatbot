import os
from textwrap import dedent
from typing import List, Dict

# load .env so OPENAI_API_KEY / OPENAI_MODEL are available
from dotenv import load_dotenv
load_dotenv()


def _build_prompt(question: str, retrieved: List[Dict]) -> str:
    snippets = "\n\n".join(f"[{r['rank']}] {r['text']}" for r in retrieved)
    return dedent(f"""
    You are TravelMate, a helpful travel assistant. Answer the user's question
    **using only** the information in the retrieved snippets below. If the answer
    is not present, say you don't have enough information. Cite sources inline
    with bracketed numbers like [1], [2]. Keep the answer concise.

    Retrieved snippets:
    {snippets}

    Question: {question}
    Answer:
    """).strip()


def _call_openai(prompt: str) -> str | None:
    """Return model output, or None if no key is set."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI
        # create client without extra kwargs; it reads env vars automatically
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(LLM error: {e})"


def generate_answer(question: str, retrieved: List[Dict]) -> str:
    prompt = _build_prompt(question, retrieved)
    llm_answer = _call_openai(prompt)
    if llm_answer is None:
        # Fallback when no API key is present: show retrieved snippets
        header = "No LLM configured — showing top retrieved snippets with citations:\n"
        body = "\n\n".join(f"[{r['rank']}] {r['text'][:500]}..." for r in retrieved)
        return header + body
    return llm_answer
