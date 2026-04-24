from __future__ import annotations

"""
Prepare extractive Wikipedia prompts for n-gram speculative decoding.

The output JSONL is intentionally prompt-ID based and tokenizer-ready so
inference scripts can focus on proposal, verification, and metrics.
"""

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

from common.tokenizer import load_tokenizer


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = "data/wiki_extract_ngram_eval100_qwen25_7b.jsonl"
DEFAULT_NUM_QUESTIONS = 100
DEFAULT_PROMPT_TOKEN_BUDGET = 14336
DEFAULT_WIKIPEDIA_TITLES = (
    "Water cycle",
    "Photosynthesis",
    "Plate tectonics",
    "Solar System",
    "Renaissance",
)
WIKIPEDIA_USER_AGENT = "speculative-decoding-ngram-benchmark/0.1"

STOP_WORDS = {
    "about",
    "after",
    "also",
    "because",
    "between",
    "during",
    "first",
    "their",
    "there",
    "these",
    "through",
    "which",
    "while",
    "where",
    "would",
}


def fetch_wikipedia_article(title: str, timeout_s: float = 30.0) -> tuple[str, str]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "titles": title,
        }
    )
    request = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    page = next(iter(payload["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"Wikipedia article not found or empty: {title}")
    return str(page.get("title", title)), str(page["extract"])


def split_wikipedia_sections(article_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Lead"
    lines: list[str] = []
    for raw_line in article_text.splitlines():
        line = raw_line.strip()
        heading_match = re.fullmatch(r"=+\s*(.*?)\s*=+", line)
        if heading_match:
            text = "\n".join(lines).strip()
            if text:
                sections.append((heading, text))
            heading = heading_match.group(1).strip() or "Section"
            lines = []
            continue
        lines.append(raw_line)
    text = "\n".join(lines).strip()
    if text:
        sections.append((heading, text))
    return sections


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalized)
    sentences = []
    for part in parts:
        sentence = part.strip()
        words = sentence.split()
        if 10 <= len(words) <= 45 and sentence.endswith((".", "!", "?")):
            sentences.append(sentence)
    return sentences


def choose_sentence_cue(sentence: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]{5,}", sentence)
    for word in sorted(words, key=len, reverse=True):
        cleaned = word.strip("-")
        if cleaned.lower() not in STOP_WORDS:
            return cleaned
    return "topic"


def build_wikipedia_question_specs(
    article_title: str,
    article_text: str,
    count: int,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for section_title, section_text in split_wikipedia_sections(article_text):
        if section_title.lower() in {"references", "external links", "see also", "notes"}:
            continue
        for sentence in split_sentences(section_text):
            if article_title.lower() in sentence.lower() or len(candidates) % 3 != 0:
                candidates.append(
                    {
                        "article_title": article_title,
                        "section_title": section_title,
                        "cue": choose_sentence_cue(sentence),
                        "expected_answer": sentence,
                    }
                )
    if len(candidates) < count:
        raise ValueError(
            f"{article_title} only produced {len(candidates)} usable extractive questions; "
            f"needed {count}"
        )

    step = max(1, len(candidates) // count)
    selected: list[dict[str, str]] = []
    used_indices: set[int] = set()
    index = 0
    while len(selected) < count and index < len(candidates):
        selected.append(candidates[index])
        used_indices.add(index)
        index += step
    if len(selected) < count:
        for index, candidate in enumerate(candidates):
            if index in used_indices:
                continue
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected[:count]


def render_extract_prompt(
    tokenizer,
    article_title: str,
    article_text: str,
    section_title: str,
    cue: str,
) -> str:
    user_content = (
        "Use the Wikipedia article below as the only source. "
        f'Quote the single exact sentence from the "{section_title}" section that contains the term "{cue}". '
        "Return only that sentence.\n\n"
        f"Article title: {article_title}\n\n"
        f"{article_text}"
    )
    messages = [
        {
            "role": "system",
            "content": "You answer extractive questions by copying exact contiguous text from the provided article.",
        },
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nAnswer:"


def prepare_dataset(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.model_path)
    titles = [title.strip() for title in args.titles.split(",") if title.strip()]
    if not titles:
        raise ValueError("--titles must contain at least one article title")

    base_count = args.num_questions // len(titles)
    remainder = args.num_questions % len(titles)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rows_written = 0
    metadata = {
        "model_path": args.model_path,
        "num_questions": args.num_questions,
        "titles": titles,
        "prompt_token_budget": args.prompt_token_budget,
        "articles": [],
    }
    with output_path.open("a", encoding="utf-8") as handle:
        for title_index, requested_title in enumerate(titles):
            article_title, article_text = fetch_wikipedia_article(requested_title)
            target_count = base_count + (1 if title_index < remainder else 0)
            specs = build_wikipedia_question_specs(article_title, article_text, target_count)
            article_prompt_lengths: list[int] = []
            for spec in specs:
                prompt = render_extract_prompt(
                    tokenizer,
                    article_title=article_title,
                    article_text=article_text,
                    section_title=spec["section_title"],
                    cue=spec["cue"],
                )
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                article_prompt_lengths.append(len(prompt_ids))
                if len(prompt_ids) > args.prompt_token_budget:
                    raise ValueError(
                        f"{article_title} prompt has {len(prompt_ids)} tokens, "
                        f"exceeding --prompt-token-budget={args.prompt_token_budget}. "
                        "Choose a smaller article or increase the budget."
                    )
                rows_written += 1
                row = {
                    "prompt_id": f"wiki_{rows_written:04d}",
                    "prompt": prompt,
                    "prompt_ids": prompt_ids,
                    "article_title": article_title,
                    "section_title": spec["section_title"],
                    "cue": spec["cue"],
                    "expected_answer": spec["expected_answer"],
                    "prompt_tokens": len(prompt_ids),
                    "prompt_truncated": False,
                    "source": "wikipedia",
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            metadata["articles"].append(
                {
                    "title": article_title,
                    "questions": target_count,
                    "article_chars": len(article_text),
                    "min_prompt_tokens": min(article_prompt_lengths) if article_prompt_lengths else 0,
                    "max_prompt_tokens": max(article_prompt_lengths) if article_prompt_lengths else 0,
                }
            )

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare extractive Wikipedia prompts for n-gram decoding.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    parser.add_argument("--titles", default=",".join(DEFAULT_WIKIPEDIA_TITLES))
    parser.add_argument("--prompt-token-budget", type=int, default=DEFAULT_PROMPT_TOKEN_BUDGET)
    return parser.parse_args(argv)


def main() -> None:
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
