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
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are TravelMate, a helpful travel assistant. Answer questions using only the provided information. Keep answers concise and cite sources."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(LLM error: {e})"


def _extract_relevant_text(question: str, chunk_text: str, max_sentences: int = 3) -> str:
    """Extract the most relevant sentences from a chunk based on the question."""
    question_words = set(word.lower() for word in question.split() if len(word) > 3)
    sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
    
    # Score sentences based on keyword matches
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for word in question_words if word in sentence_lower)
        # Bonus for important keywords
        if any(kw in sentence_lower for kw in ['cost', 'price', 'euro', 'eur', 'usd', '$', '€', 'pound', 'yen', 'baht', 'dollar']):
            score += 2
        if any(kw in sentence_lower for kw in ['airport', 'train', 'bus', 'taxi', 'metro', 'subway']):
            score += 1
        scored_sentences.append((score, sentence))
    
    # Sort by relevance score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    relevant = [s for score, s in scored_sentences[:max_sentences] if score > 0]
    
    if relevant:
        return '. '.join(relevant) + '.'
    # Fallback: return first few sentences
    return '. '.join(sentences[:max_sentences]) + '.'

def generate_answer(question: str, retrieved: List[Dict]) -> str:
    prompt = _build_prompt(question, retrieved)
    llm_answer = _call_openai(prompt)
    if llm_answer is None:
        # Fallback when no API key is present: synthesize answer from most relevant chunks
        if not retrieved:
            return "I don't have enough information to answer that question."
        
        # Use only the most relevant chunks
        relevant_chunks = retrieved[:3]
        
        # Extract only relevant information from chunks based on the question
        answer_parts = []
        sources_used = set()
        
        for chunk in relevant_chunks:
            # Extract only sentences relevant to the question
            extracted = _extract_relevant_text(question, chunk['text'])
            
            if extracted and len(extracted.strip()) > 20:  # Only add if we have substantial content
                answer_parts.append(extracted)
                sources_used.add(chunk['source'])
                
                # Stop if we have enough information
                if len(answer_parts) >= 2:
                    break
        
        if answer_parts:
            answer = ' '.join(answer_parts)
            # Add source citation
            if sources_used:
                if len(sources_used) == 1:
                    answer += f"\n\n(Source: {list(sources_used)[0]})"
                else:
                    answer += f"\n\n(Sources: {', '.join(list(sources_used)[:2])})"
            return answer
        else:
            # Final fallback: use most relevant chunk
            best_chunk = relevant_chunks[0]
            relevant_text = _extract_relevant_text(question, best_chunk['text'], max_sentences=3)
            if len(relevant_text) > 20:
                answer = relevant_text
            else:
                answer = best_chunk['text'][:250] + "..."
            answer += f"\n\n(Source: {best_chunk['source']})"
            return answer
    return llm_answer
