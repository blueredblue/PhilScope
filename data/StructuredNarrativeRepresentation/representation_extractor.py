import os
import json
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from litellm import completion

# =====================================================================
# 0. DIRECTORY LAYOUT
# =====================================================================
# This file lives at: data/StructuredNarrativeRepresentation/representation_extractor.py
# Sibling corpora:
#   data/SourcePhilosophyEssays/extracted_text/  -> human-written essays (.txt)
#   data/SyntheticEssays/gen_synthetic_essays/    -> LLM-generated essays (.txt, filenames end in the generating model's name)
#
# All outputs land in one shared directory so everything for a given source
# essay sorts together:
#   data/StructuredNarrativeRepresentation/structured_representations/
#
# Provenance is preserved two ways even in a flat folder:
#   - filename: human essays get "_human" appended (synthetic filenames already
#     end in the model name, e.g. "..._gpt-5.6-luna_representation.json")
#   - JSON content: every output is stamped with "_source_corpus" and "_source_file"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent

# API keys (e.g. OPENROUTER_API_KEY, GEMINI_API_KEY) live in SyntheticEssays/.env.
# Load them explicitly since Python doesn't read .env files on its own -
# without this, litellm silently has no credentials and every call fails auth.
ENV_PATH = DATA_DIR / "SyntheticEssays" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"[!] Warning: expected .env at '{ENV_PATH}' but it wasn't found. "
          f"Model API calls will fail unless credentials are already set in your shell environment.")

CORPORA = {
    "human": DATA_DIR / "SourcePhilosophyEssays" / "extracted_text",
    "synthetic": DATA_DIR / "SyntheticEssays" / "gen_synthetic_essays",
}

# =====================================================================
# 1. PHILOSOPHICAL ESSAY EXTRACTION SCHEMA (PYDANTIC)
# =====================================================================

class ArgumentMacroStructure(BaseModel):
    dominant_argument_form: str = Field(
        description="Core logical form (e.g., Deductive, Inductive, Reductio ad absurdum, Inference to the best explanation, Transcendental)."
    )
    premise_explicitness: str = Field(
        description="Degree of explicitness (e.g., Fully explicit & formal, Enthymematic/implicit assumptions present)."
    )
    num_distinct_arguments_for_thesis: int = Field(
        description="Count of distinct independent arguments supporting the main thesis."
    )

class DialecticalEngagement(BaseModel):
    objections_raised: List[str] = Field(
        description="List of specific counterarguments or objections brought up in the essay."
    )
    objection_handling: str = Field(
        description="Method of resolving objections (e.g., Rebuttal of premises, Distinction of terms, Re-framing, Dismissal)."
    )
    bites_the_bullet_on_counterintuitive_implication: bool = Field(
        description="True if the author accepts a counterintuitive implication of their view rather than denying it."
    )

class MethodAndEvidence(BaseModel):
    uses_named_thought_experiment: bool = Field(
        description="True if a named or explicit thought experiment is introduced."
    )
    named_thought_experiments: List[str] = Field(
        default=[],
        description="Names or brief labels of thought experiments used (e.g., 'Trolley Problem', 'Mary's Room')."
    )
    evidence_type: str = Field(
        description="Primary evidence mode (e.g., Conceptual analysis, Empirical data, Intuition pumps, Formal logic, Phenomenological analysis)."
    )
    appeals_to_authority: List[str] = Field(
        default=[],
        description="Names of explicit historical or contemporary philosophers cited/referenced."
    )

class ThesisCommitment(BaseModel):
    conclusion_type: str = Field(
        description="Character of the final thesis (e.g., Definitive assertion, Conditional claim, Aporia/Skeptical stance, Exploratory suggestion)."
    )
    hedging_density_in_final_paragraph: str = Field(
        description="Frequency of epistemic hedges (e.g., 'perhaps', 'seems', 'might') in the conclusion: High, Medium, or Low."
    )

class StructuralOrganization(BaseModel):
    structure_type: str = Field(
        description="Architectural layout (e.g., Traditional Tripartite, Dialectical/Hegelian, Problem-Solution, Sectional Taxonomy)."
    )
    explicit_signposting: bool = Field(
        description="True if explicit organizational meta-language is used (e.g., 'First, I will argue...', 'In Section 2...')."
    )

class EpistemicStanceAndVoice(BaseModel):
    certainty_language_frequency: str = Field(
        description="Usage of high-certainty indicators (e.g., 'clearly', 'must', 'necessarily', 'indisputable'): High, Medium, or Low."
    )
    uses_first_person_epistemic_verbs: bool = Field(
        description="True if author uses first-person stance verbs ('I argue', 'I contend', 'I propose')."
    )
    imagined_interlocutor_voiced_directly: bool = Field(
        description="True if an opposing voice or critic is given direct quotation or dialogue-like phrasing."
    )

class ConceptualGrounding(BaseModel):
    explicit_definitions_given: List[str] = Field(
        default=[],
        description="Key philosophical terms explicitly defined with technical precision."
    )
    draws_explicit_distinction: bool = Field(
        description="True if the text explicitly draws a distinction (e.g., 'X in sense A vs. sense B')."
    )
    distinctions_drawn: List[str] = Field(
        default=[],
        description="Summary of explicit distinctions drawn."
    )

class ArgumentativeQuality(BaseModel):
    premise_acceptability: str = Field(
        description="Assessment of premise grounding (e.g., Intuitively sound, Empirically supported, Controversial, Unsubstantiated)."
    )
    logical_relevance: str = Field(
        description="Do the premises validly or cogently entail the conclusion? (Valid entailment, Moderate gap, Non-sequitur)."
    )
    dialectical_reasonableness: str = Field(
        description="Is objection handling fair and fatal, or straw-manned/superficial?"
    )

class PhilosophicalOrientation(BaseModel):
    thematic_focus: str = Field(
        description="Primary philosophical sub-discipline (e.g., Epistemology, Metaphysics, Ethics, Philosophy of Mind, Political Philosophy)."
    )
    sentiment: str = Field(
        description="Core posture: 'Critical-Destructive' (dismantling an existing view) or 'Constructive-System Building' (defending a positive model)."
    )
    grounding: str = Field(
        description="Methodological orientation: 'Formal' (symbolic/structural logic) vs. 'Intuitive' (conceptual/ordinary language)."
    )

# --- MASTER BLUEPRINT CONTAINER ---
class PhilosophyEssayBlueprint(BaseModel):
    argument_macro_structure: ArgumentMacroStructure
    dialectical_engagement: DialecticalEngagement
    method_and_evidence: MethodAndEvidence
    thesis_commitment: ThesisCommitment
    structural_organization: StructuralOrganization
    epistemic_stance_and_voice: EpistemicStanceAndVoice
    conceptual_grounding: ConceptualGrounding
    argumentative_quality: ArgumentativeQuality
    philosophical_orientation: PhilosophicalOrientation
    summary_of_core_decisions: str = Field(
        description="3-4 sentence high-level executive summary of the paper's overarching argumentative strategy."
    )

# =====================================================================
# 2. EXTRACTION SYSTEM PROMPT
# =====================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert academic philosophy referee and analytical taxonomist.

Your task is to analyze the provided philosophical text and extract its structural, dialectical, and methodological blueprint across 9 core dimensions.

Focus purely on the underlying logical architecture, argumentative choices, and conceptual mechanics.
Do NOT output surface stylistic summaries or copy large blocks of prose. 
You must strictly conform your output to the provided JSON schema.
"""

# =====================================================================
# 3. EXTRACTION PIPELINE
# =====================================================================

def extract_representation(essay_text: str, model_id: str, retries: int = 3) -> Optional[dict]:
    """Passes raw essay text to LLM and returns structured JSON conforming to the Pydantic schema."""
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the following philosophical text and extract its structural blueprint:\n\n{essay_text}"}
    ]

    for attempt in range(retries):
        try:
            response = completion(
                model=model_id,
                messages=messages,
                response_format=PhilosophyEssayBlueprint,
                temperature=0.1  # Low temperature for deterministic analysis
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"   [!] Extraction attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return None


def _process_single_file(
    filename: str,
    corpus: str,
    raw_dir_path: Path,
    output_dir_path: Path,
    extractor_model: str,
) -> str:
    """Does the work for one essay: read -> extract -> save. Returns a short status string for logging."""
    base_name = os.path.splitext(filename)[0]
    if corpus == "human":
        base_name = f"{base_name}_human"
    output_filepath = output_dir_path / f"{base_name}_representation.json"

    # Skip if already processed (crash/resume safety)
    if output_filepath.exists():
        return f"SKIP: '{filename}' (already exists)"

    with open(raw_dir_path / filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    structured_data = extract_representation(raw_text, model_id=extractor_model)

    if structured_data:
        # Tag provenance so human vs. synthetic origin is never lost downstream
        structured_data["_source_corpus"] = corpus
        structured_data["_source_file"] = filename

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2)
        return f"OK: '{filename}' -> '{output_filepath.name}'"
    else:
        return f"FAIL: '{filename}'"


def process_stage_1(
    corpus: str = "human",
    extractor_model: str = "gemini/gemini-2.5-flash",
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_workers: int = 5,
):
    """
    Iterates through all raw text files for a given corpus ('human' or 'synthetic')
    and produces JSON representation blueprints, tagged with their source corpus.

    raw_dir / output_dir can be passed explicitly to override the CORPORA defaults
    (e.g. if you add a third corpus later).

    max_workers controls how many essays are extracted concurrently. Each extraction
    is I/O-bound (waiting on an API response), so running several at once mostly just
    overlaps that waiting time - it does not change the number of API calls made or
    their cost, only the wall-clock time to finish. If you're on a low-tier/free API
    key and start seeing rate-limit errors, lower this number (e.g. 2-3).
    """
    if raw_dir is None:
        if corpus not in CORPORA:
            print(f"Error: unknown corpus '{corpus}'. Expected one of {list(CORPORA)} or pass raw_dir explicitly.")
            return
        raw_dir_path = CORPORA[corpus]
    else:
        raw_dir_path = Path(raw_dir)

    if output_dir is None:
        output_dir_path = SCRIPT_DIR / "structured_representations"
    else:
        output_dir_path = Path(output_dir)

    if not raw_dir_path.exists():
        print(f"Error: Raw essays directory '{raw_dir_path}' does not exist.")
        return

    output_dir_path.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(f for f in os.listdir(raw_dir_path) if f.endswith(".txt"))
    total_files = len(raw_files)

    print(f"Starting Stage 1: Processing {total_files} '{corpus}' essays using model '{extractor_model}' "
          f"(up to {max_workers} concurrent requests)...")
    print(f"  Source: {raw_dir_path}")
    print(f"  Output: {output_dir_path}")
    print("=" * 70)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_file, filename, corpus, raw_dir_path, output_dir_path, extractor_model
            ): filename
            for filename in raw_files
        }

        for future in as_completed(futures):
            completed += 1
            filename = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = f"FAIL: '{filename}' (unexpected error: {e})"

            if result.startswith("OK"):
                print(f"[{completed}/{total_files}] [\u2713] {result[4:]}")
            elif result.startswith("SKIP"):
                print(f"[{completed}/{total_files}] {result}")
            else:
                print(f"[{completed}/{total_files}] [X] {result[6:]}")

    print(f"\nStage 1 Complete for corpus '{corpus}'! Structured representations saved in '{output_dir_path}'.")


if __name__ == "__main__":
    # Run once per corpus. Change model_id here if you want to swap providers/models
    # (litellm accepts e.g. "gpt-4o", "anthropic/claude-sonnet-4-6", "gemini/gemini-2.5-flash", etc.)
    MODEL_ID = "openrouter/deepseek/deepseek-v4-flash-0731:floor"

    # How many essays to extract concurrently. Higher = faster wall-clock time,
    # same total API cost. Lower this if you start seeing rate-limit errors.
    MAX_WORKERS = 5

    process_stage_1(corpus="human", extractor_model=MODEL_ID, max_workers=MAX_WORKERS)
    process_stage_1(corpus="synthetic", extractor_model=MODEL_ID, max_workers=MAX_WORKERS)