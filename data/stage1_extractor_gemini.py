import os
import glob
import json
import time
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize the Gemini Client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()

# Choose your model:
# - "gemini-2.5-flash" (or "gemini-1.5-flash"): Extremely fast, cheap/free, and highly capable
# - "gemini-2.5-pro" (or "gemini-1.5-pro"): Deep reasoning power for complex academic prose
GEMINI_MODEL = "gemini-2.5-flash"

# =====================================================================
# 1. PYDANTIC SCHEMA DEFINITIONS (The 11 Philosophical Dimensions)
# =====================================================================

class ThesisCommitmentClaim(BaseModel):
    conclusion_summary: str = Field(description="The exact main philosophical conclusion or claim the essay asserts.")
    commitment_stance_description: str = Field(description="A qualitative analysis of how the author commits to this thesis (tone, conditionality, conviction).")

class GroundsData(BaseModel):
    explicit_premises: str = Field(description="A detailed breakdown of the factual assertions, empirical data, or premises explicitly stated.")
    implicit_assumptions: str = Field(description="Analysis of unstated assumptions or enthymemes the argument relies on.")

class Warrant(BaseModel):
    inferential_bridge: str = Field(description="The logical principle or rule connecting the grounds to the thesis.")
    explicitness_mechanics: str = Field(description="How this bridge is delivered (e.g., defended, assumed, or hidden).")

class Backing(BaseModel):
    warrant_justification: str = Field(description="How the author justifies the warrant itself (e.g., intuition, physics, axioms).")
    intellectual_genealogy: str = Field(description="The thinkers, texts, or historical schools cited to support the warrant.")

class QualifierEpistemicStance(BaseModel):
    epistemic_modality: str = Field(description="Analysis of modal limiting language (e.g., 'necessarily' vs 'probably').")
    scope_parameters: str = Field(description="The qualitative boundaries or limits drawn around the thesis.")

class Rebuttal(BaseModel):
    objection_integration: str = Field(description="Description of the counterarguments introduced and their placement.")
    handling_and_defense_strategy: str = Field(description="How the author structurally resolves or concedes to those objections.")

class Method(BaseModel):
    methodological_toolkit: str = Field(description="Inventory of methods used (e.g., thought experiments, formal logic, semantic analysis).")
    operative_normative_framework: str = Field(description="The framework governing the reasoning (e.g., deontic, utilitarian).")

class PhilosophicalOrientation(BaseModel):
    thematic_focus_context: str = Field(description="The specific historical/philosophical problem or debate entered.")
    evaluative_sentiment: str = Field(description="The author's posture (e.g., critique, advocacy, synthesis).")
    grounding_methodology: str = Field(description="How abstract ideas are anchored (e.g., raw concepts vs. concrete examples).")

class Cogency(BaseModel):
    local_logic_mechanics: str = Field(description="Evaluation of internal logical consistency, transitions, loops, or fallacies.")

class Reasonableness(BaseModel):
    dialectical_depth: str = Field(description="Analysis of dialectical depth, fair handling of opponent views, or steelmanning.")

class Effectiveness(BaseModel):
    rhetorical_and_prose_execution: str = Field(description="Analysis of the stylistic, structural, and rhetorical strategies used to deliver the logic.")

# Master Class
class PhilosophyEssayBlueprint(BaseModel):
    thesis_commitment_claim: ThesisCommitmentClaim
    grounds_data: GroundsData
    warrant: Warrant
    backing: Backing
    qualifier_epistemic_stance: QualifierEpistemicStance
    rebuttal: Rebuttal
    method: Method
    philosophical_orientation: PhilosophicalOrientation
    cogency: Cogency
    reasonableness: Reasonableness
    effectiveness: Effectiveness

# =====================================================================
# 2. SYSTEM PROMPT
# =====================================================================

SYSTEM_PROMPT = """You are a Senior Philosophical Analyst and Argumentation Extraction Engine. 

Your sole task is to ingest raw philosophical essays and dissect their underlying logical, dialectical, and rhetorical architecture. You must strip away stylistic noise, aesthetic word choices, and purely ornamental prose to map the "load-bearing" structure of the argument.

To ensure your extraction is rigorous and valuable for downstream comparative analysis, adhere strictly to these principles:

1. OBJECTIVITY & DESCRIPTIVE RIGOR: Analyze the text exactly as written. Do not attempt to fix the author's argument, steelman a weak premise unless the text itself attempts to do so, or project external philosophical concepts onto the text.
2. TOULMIN-MODEL ALIGNMENT: 
   - Pay close attention to how the "Grounds" support the "Thesis/Claim".
   - Identify the "Warrant"—the often-unstated bridge of inference that makes the jump from Grounds to Claim logically permissible.
   - Look for the "Backing" (how the author defends that warrant) and "Qualifiers" (how the author limits the certainty of their conclusion).
3. EXPLICIT VS. IMPLICIT: Distinguish sharply between what the author says outright (explicit) and the foundational, unstated assumptions they are quietly leaning on (implicit).
4. METHODOLOGICAL FOCUS: Isolate the specific tools the author uses to move the dialectic forward (e.g., thought experiments, conceptual analysis, formal deductions).
5. LOGICAL AUDITING: When analyzing "Cogency" and "Reasonableness", do not merely summarize; audit the mechanics. Point out leaps in logic, non-sequiturs, or areas where opposing viewpoints are unfairly trivialized.

You must output your analysis strictly conforming to the provided JSON Schema."""

# =====================================================================
# 3. GEMINI EXTRACTION PIPELINE WITH CACHING & RETRIES
# =====================================================================

def extract_stage1_blueprint_gemini(essay_text: str, retries: int = 3) -> Optional[PhilosophyEssayBlueprint]:
    """Sends essay text to Gemini API enforcing the Pydantic schema via response_schema."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Analyze this philosophical essay:\n\n{essay_text}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=PhilosophyEssayBlueprint,
                    temperature=0.1,  # Low temperature for analytical consistency
                ),
            )
            
            # Use response.parsed if populated, otherwise parse the JSON string explicitly
            if hasattr(response, "parsed") and response.parsed:
                return response.parsed
            else:
                return PhilosophyEssayBlueprint.model_validate_json(response.text)

        except Exception as e:
            print(f"   [!] Error on attempt {attempt + 1}: {e}")
            time.sleep(2)
    return None

def process_all_essays(input_dir: str = "raw_essays", output_dir: str = "stage1_blueprints"):
    """Iterates over all .txt files, extracts Stage 1 JSONs, and saves them."""
    os.makedirs(output_dir, exist_ok=True)
    text_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not text_files:
        print(f"No .txt files found in '{input_dir}'. Please add your essays there.")
        return

    print(f"Found {len(text_files)} essays to process using Gemini ({GEMINI_MODEL}).\n" + "="*50)

    for idx, filepath in enumerate(text_files, 1):
        filename = os.path.basename(filepath)
        json_filename = os.path.splitext(filename)[0] + ".json"
        output_filepath = os.path.join(output_dir, json_filename)

        # IDEMPOTENCY: Skip files that have already been extracted
        if os.path.exists(output_filepath):
            print(f"[{idx}/{len(text_files)}] Skipping '{filename}' (JSON already exists).")
            continue

        print(f"[{idx}/{len(text_files)}] Extracting structural blueprint for: '{filename}'...")
        
        with open(filepath, "r", encoding="utf-8") as f:
            essay_text = f.read()

        parsed_blueprint = extract_stage1_blueprint_gemini(essay_text)

        if parsed_blueprint:
            # Convert Pydantic object to dictionary and save as JSON
            blueprint_dict = parsed_blueprint.model_dump()
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(blueprint_dict, f, indent=2)
            print(f"   [✓] Saved blueprint to '{output_filepath}'")
        else:
            print(f"   [X] Failed to extract blueprint for '{filename}' after multiple retries.")

    print("\n" + "="*50 + "\nStage 1 Processing Complete!")

if __name__ == "__main__":
    process_all_essays()