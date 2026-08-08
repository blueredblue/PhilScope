import os
import glob
import json
import time
import argparse
from google import genai

client = genai.Client()

STAGE_0_PROMPT = """

You are a philosophy instructor generating **essay prompts** based on published philosophical papers.

For the essay below, craft one prompt that:
1. Begins with **"Write a philosophical essay"** (exact phrase).
2. Continues the sentence with **"exploring...", "arguing that...", "evaluating...", "critiquing...", or "addressing the question of..."** and introduces at least one key philosophical concept, central thesis, or named thinker/thought experiment directly from the text.
3. Conveys the essay's distinctive **philosophical core / problem / normative domain**, providing a clear sense of the dialectical objective, and includes **concrete elements** (e.g., central mechanism, specific objection, thought experiment, or theoretical framework)—don't overload with minutiae.
4. Offers enough argumentative guidance to get the writer started (a clear philosophical tension, dilemma, or question to resolve) yet leaves room for their own reasoning, counterarguments, and conclusions.
5. Do **not** address the reader in second person; keep the prompt in third-person imperative or objective phrasing (no "you/your").
6. Avoid vague hedge words (*maybe*, *perhaps*, *consider*) **and** absolutely do NOT use comparison phrases or qualifiers such as *like*, *much like*, *similar to*, *reminiscent of*, *in the style of*. Refer to the concrete concepts, thinkers, or arguments directly. Do not invent philosophical terms, thought experiments, or author names that do not appear in the text.
7. Single paragraph <= 120 words.

Return ONLY the prompt text—**no extra commentary**. If anything else is included, keep it on the same line separated by a single space.

ESSAY TO ANALYSE:
{essay_text}
"""  # Paste your full prompt text here — keep the {essay_text} placeholder


def load_existing(output_file: str) -> dict:
    """Load already-generated prompts so re-runs skip completed essays."""
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_dataset(prompts_dataset: dict, output_file: str):
    """Write the current state of the dataset to disk."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(prompts_dataset, f, indent=2)


def generate_prompt_for_essay(essay_text: str, model: str, retries: int = 3) -> str | None:
    """Call Gemini to infer the prompt behind an essay, with retries."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=STAGE_0_PROMPT.format(essay_text=essay_text),
                config={"temperature": 0.2},
            )
            return response.text.strip()
        except Exception as e:
            print(f"   [!] Error on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def extract_prompts(
    source_dir: str = "data/SourcePhilosophyEssays/extracted_text/",
    output_file: str = "data/BenchmarkPrompts/benchmark_prompts.json",
    model: str = "gemini-3.5-flash-lite",
):
    essay_files = glob.glob(os.path.join(source_dir, "*.txt"))

    if not essay_files:
        print(f"No .txt files found in '{source_dir}'. Nothing to do.")
        return

    prompts_dataset = load_existing(output_file)
    already_done = set(prompts_dataset.keys())

    print(f"Found {len(essay_files)} source essays. "
          f"{len(already_done)} already processed, "
          f"{len(essay_files) - len(already_done)} remaining.\n" + "=" * 50)

    for idx, filepath in enumerate(essay_files, 1):
        filename = os.path.basename(filepath)
        essay_id = os.path.splitext(filename)[0]

        # IDEMPOTENCY: skip essays we've already generated a prompt for
        if essay_id in already_done:
            print(f"[{idx}/{len(essay_files)}] Skipping '{filename}' (already in dataset).")
            continue

        print(f"[{idx}/{len(essay_files)}] Generating prompt for: '{filename}'...")

        with open(filepath, "r", encoding="utf-8") as f:
            essay_text = f.read()

        generated_prompt = generate_prompt_for_essay(essay_text, model=model)

        if generated_prompt:
            # Programmatically append the word count requirement
            word_count_suffix = "Your essay must be between 3200 and 5200 words long."
            if not generated_prompt.strip().endswith(word_count_suffix):
                generated_prompt = generated_prompt.rstrip() + " " + word_count_suffix

            prompts_dataset[essay_id] = {
                "source_file": filename,
                "extracted_prompt": generated_prompt,
            }
            # Save after every essay so a crash never loses completed work
            save_dataset(prompts_dataset, output_file)
            print(f"   [✓] Saved. Prompt:\n       \"{generated_prompt}\"\n")
        else:
            print(f"   [X] Failed to generate prompt for '{filename}' after retries. Skipping for now.\n")

    print("=" * 50 + f"\nDone. {len(prompts_dataset)}/{len(essay_files)} essays have prompts in '{output_file}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark prompts from source essays via Gemini.")
    parser.add_argument(
        "--source-dir", "-s",
        default="data/SourcePhilosophyEssays/extracted_text/",
        help="Folder containing source .txt essays (default: data/SourcePhilosophyEssays/extracted_text/)",
    )
    parser.add_argument(
        "--output-file", "-o",
        default="data/BenchmarkPrompts/benchmark_prompts.json",
        help="Path to write the resulting JSON dataset (default: data/BenchmarkPrompts/benchmark_prompts.json)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-3.5-flash-lite",
        help="Gemini model to use (default: gemini-3.5-flash-lite)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_prompts(source_dir=args.source_dir, output_file=args.output_file, model=args.model)