import os
import json
import time
import threading
from typing import List, Dict
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass

from litellm import completion



# 1. MODEL CONFIGURATION

MODELS_TO_RUN: List[Dict[str, str]] = [
    {"model_id": "gemini/gemini-3.5-flash-lite", "label": "gemini-3.5-flash-lite"},
    {"model_id": "groq/llama-3.3-70b-versatile",  "label": "llama33_70b_groq"},
    {"model_id": "openrouter/inclusionai/ling-3.0-flash:floor", "label": "ling-3.0-flash-free"},
    {"model_id": "openrouter/openai/gpt-5.6-luna:floor", "label": "gpt-5.6-luna"},
    {"model_id": "openrouter/deepseek/deepseek-v4-flash-0731:floor", "label": "deepseek-v4-flash-0731"},
]

# 2. RATE LIMITING (per provider, requests-per-minute)

# Free-tier limits — adjust to match your actual plan.
RATE_LIMITS_RPM: Dict[str, int] = {
    "gemini": 15,
    "groq": 15,
    "openrouter": 100,
}
DEFAULT_RPM = 15  # fallback for any provider not listed above


class RateLimiter:
    """
    Simple per-provider rate limiter. Before each call, blocks just long enough
    to keep requests spaced at (60 / rpm) seconds apart for that provider.
    Thread-safe in case this is ever parallelized later.
    """
    def __init__(self, limits: Dict[str, int], default_rpm: int):
        self.limits = limits
        self.default_rpm = default_rpm
        self.last_call_time: Dict[str, float] = {}
        self.lock = threading.Lock()

    def _provider_of(self, model_id: str) -> str:
        # litellm model ids are "provider/rest-of-name" -> take the first segment
        return model_id.split("/")[0]

    def wait(self, model_id: str):
        provider = self._provider_of(model_id)
        rpm = self.limits.get(provider, self.default_rpm)
        min_interval = 60.0 / rpm

        with self.lock:
            now = time.time()
            last = self.last_call_time.get(provider, 0.0)
            elapsed = now - last
            if elapsed < min_interval:
                sleep_for = min_interval - elapsed
                print(f"        [~] Rate limit ({provider}: {rpm} rpm) — waiting {sleep_for:.1f}s")
                time.sleep(sleep_for)
            self.last_call_time[provider] = time.time()


rate_limiter = RateLimiter(RATE_LIMITS_RPM, DEFAULT_RPM)


# 3. FAILURE LOG HELPERS

def load_failures(failures_file: str) -> dict:
    """Load the persistent failure log, if one already exists."""
    if os.path.exists(failures_file):
        with open(failures_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_failures(failures: dict, failures_file: str):
    """Write the current failure log to disk."""
    with open(failures_file, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)


def record_failure(failures: dict, failures_file: str, essay_id: str, model_label: str, error: str):
    """Add/update one failure entry and persist immediately."""
    key = f"{essay_id}_{model_label}"
    failures[key] = {
        "essay_id": essay_id,
        "model_label": model_label,
        "last_error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_failures(failures, failures_file)


def clear_failure(failures: dict, failures_file: str, essay_id: str, model_label: str):
    """Remove a failure entry once that pair succeeds (e.g. on a re-run)."""
    key = f"{essay_id}_{model_label}"
    if key in failures:
        del failures[key]
        save_failures(failures, failures_file)


# 4. ZERO-SHOT GENERATION (STORYSCOPE METHODOLOGY)

def generate_essay_unsteered(prompt_text: str, model_id: str, retries: int = 3) -> tuple[str, str]:
    """
    Sends ONLY the benchmark user prompt to the LLM (no system prompt),
    allowing the model's default structural idiosyncrasies to emerge.

    Returns (essay_text, last_error). essay_text is "" on total failure,
    and last_error holds the final exception message for logging.
    """
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    last_error = ""
    for attempt in range(retries):
        try:
            rate_limiter.wait(model_id)
            response = completion(
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip(), ""
        except Exception as e:
            last_error = str(e)
            print(f"   [!] Error on attempt {attempt + 1} with {model_id}: {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return "", last_error


# 5. ORCHESTRATION

def process_benchmark_prompts(
    prompts_file: str = "data/BenchmarkPrompts/benchmark_prompts.json",
    output_dir: str = "gen_synthetic_essays",
    failures_file: str = "generation_failures.json",
    max_prompts: int = None,
):
    """
    Reads benchmark_prompts.json and generates synthetic unsteered essays.
    Failures are logged persistently to failures_file so they can be
    reviewed or retried later without re-scanning every essay.

    max_prompts: if set, only process the first N prompts (handy for test runs).
    """
    if not os.path.exists(prompts_file):
        print(f"Error: Prompt file '{prompts_file}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    if max_prompts is not None:
        prompts_data = dict(list(prompts_data.items())[:max_prompts])

    failures = load_failures(failures_file)

    total_prompts = len(prompts_data)
    print(f"Loaded {total_prompts} benchmark prompts.")
    print(f"Models scheduled to run: {[m['label'] for m in MODELS_TO_RUN]}")
    print(f"Rate limits (rpm): {RATE_LIMITS_RPM} (default {DEFAULT_RPM})")
    if failures:
        print(f"({len(failures)} previously failed generations logged in '{failures_file}')")
    print("=" * 60)

    for prompt_idx, (essay_id, data) in enumerate(prompts_data.items(), 1):
        prompt_text = data["extracted_prompt"]
        print(f"[{prompt_idx}/{total_prompts}] Processing Prompt ID: '{essay_id}'")
        print(f"    Prompt snippet: \"{prompt_text[:80]}...\"")

        for model_info in MODELS_TO_RUN:
            model_id = model_info["model_id"]
            model_label = model_info["label"]
            output_filename = f"{essay_id}_{model_label}.txt"
            output_filepath = os.path.join(output_dir, output_filename)

            # Skip if file already exists
            if os.path.exists(output_filepath):
                print(f"    └── [{model_label}] Skipping (Already generated).")
                continue

            print(f"    └── [{model_label}] Generating essay...")

            essay_text, error = generate_essay_unsteered(prompt_text, model_id=model_id)

            if essay_text:
                word_count = len(essay_text.split())
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(essay_text)
                print(f"        [✓] Saved {word_count} words -> '{output_filename}'")
                # If this pair had failed on a previous run, clear it now that it succeeded
                clear_failure(failures, failures_file, essay_id, model_label)
            else:
                print(f"        [X] Failed to generate essay for '{essay_id}' using {model_label}")
                record_failure(failures, failures_file, essay_id, model_label, error)

        print("-" * 60)

    if failures:
        print(f"\nGeneration complete with {len(failures)} failure(s) logged in '{failures_file}'.")
        print("Re-run this script to retry them (already-succeeded files are skipped automatically).")
    else:
        print("\nGeneration Complete! All synthetic essays saved, no failures logged.")


if __name__ == "__main__":
    # For a quick test on just 3 prompts, use:
 #   process_benchmark_prompts(max_prompts=3, output_dir="gen_synthetic_essays_test")
     process_benchmark_prompts()