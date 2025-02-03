#!/usr/bin/env python3
import os
import sys
import json
import openai
import time
import mimetypes
import getpass
from pathlib import Path
from pdfminer.high_level import extract_text
from dotenv import dotenv_values, set_key
import argparse
import openai.error

# Global variable for OpenAI API retry attempts (can be overridden via CLI)
API_MAX_RETRIES = 3

# ---------------------------
# Environment & API Key Setup
# ---------------------------
def ensure_env_file(env_path):
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")

def save_api_key(api_key, env_path):
    set_key(str(env_path), "OPENAI_API_KEY", api_key)

def load_api_key(env_path):
    # Removed printing of environment contents for security.
    env_config = dotenv_values(str(env_path))
    return env_config.get("OPENAI_API_KEY")

def validate_api_key(api_key):
    return bool(api_key and api_key.strip().startswith("sk-"))

def test_api_key(api_key):
    """Test the API key by attempting to list models."""
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False
    except Exception as e:
        print(f"Warning: Unable to verify API key: {e}")
        return False

def prompt_for_api_key():
    return getpass.getpass("Enter your OpenAI API key: ").strip()

def get_api_key(env_path):
    """Retrieve and validate the OpenAI API key, allowing up to 3 attempts."""
    env_api_key = os.environ.get("OPENAI_API_KEY")
    if validate_api_key(env_api_key) and test_api_key(env_api_key):
        return env_api_key

    cached_key = load_api_key(env_path)
    if validate_api_key(cached_key) and test_api_key(cached_key):
        return cached_key

    for _ in range(3):
        new_key = prompt_for_api_key()
        if validate_api_key(new_key) and test_api_key(new_key):
            save_api_key(new_key, env_path)
            return new_key
        print("Invalid API key. Please try again.")

    print("Failed to provide a valid OpenAI API key after multiple attempts. Exiting.")
    sys.exit(1)

# ---------------------------
# Helper Functions for LLM Queries
# ---------------------------
def clean_code_block(text):
    """
    If the text is wrapped in a markdown code block (e.g., ```json ... ```),
    strip out the code block markers.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().endswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text

def fix_truncated_json(content):
    """
    Attempt to fix a truncated JSON string by finding the last '}' and
    taking the substring up to that point.
    """
    last_brace = content.rfind("}")
    if last_brace != -1:
        candidate = content[:last_brace+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

def attempt_fix_json(content):
    """
    Try to heuristically fix JSON that appears to be truncated by replacing
    newlines with spaces and appending missing '}' characters (up to 10 times).
    """
    fixed_content = content.replace("\n", " ").strip()
    open_braces = fixed_content.count("{")
    close_braces = fixed_content.count("}")
    attempts = 0
    while open_braces > close_braces and attempts < 10:
        fixed_content += "}"
        close_braces = fixed_content.count("}")
        attempts += 1
    try:
        return json.loads(fixed_content)
    except Exception:
        return None

def call_openai_with_retries(model, messages, max_retries=None, base_delay=2, max_tokens=500):
    """Call OpenAI API with retry logic for rate limits and transient errors."""
    if max_retries is None:
        max_retries = API_MAX_RETRIES
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.RateLimitError:
            wait_time = base_delay * (2 ** attempt)
            print(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except (openai.error.APIError, openai.error.Timeout) as e:
            print(f"OpenAI API error: {e}. Retrying in {base_delay} seconds...")
            time.sleep(base_delay)
        except Exception as e:
            print(f"Unexpected error during API call: {e}. Aborting request.")
            return None
    print("Max retries reached. Skipping this request.")
    return None

def query_concept_status(segment):
    """
    Given a text segment, ask the LLM if it is:
      - "complete": a single complete conceptual unit,
      - "multiple": containing more than one distinct concept, or
      - "partial": still an incomplete fragment.
    The LLM must reply with exactly one word: complete, multiple, or partial.
    """
    prompt = f"""Below is a text segment extracted from a document.
Determine if it represents a complete conceptual unit (a single, self-contained idea),
if it contains multiple distinct conceptual units, or if it is an incomplete fragment.
Respond with exactly one word: complete, multiple, or partial. Do not include any additional text.

Segment:
{segment}
"""
    messages = [
        {"role": "system", "content": "You are an expert at analyzing text segments for conceptual completeness."},
        {"role": "user", "content": prompt}
    ]
    answer = call_openai_with_retries(model="gpt-4", messages=messages, max_retries=API_MAX_RETRIES, max_tokens=10)
    answer = answer.strip().lower() if answer else ""
    if answer in ["complete", "multiple", "partial"]:
        return answer
    return "complete"

def split_multiple_concepts(segment):
    """
    Given a segment that the LLM indicates contains multiple concepts,
    ask the LLM to split it into a JSON object with two keys:
      - "units": an array of strings, each being one complete conceptual unit.
      - "remainder": a string containing any leftover text that does not form a complete unit.
    This function retries up to API_MAX_RETRIES times if parsing fails.
    """
    prompt = f"""The following text segment contains multiple distinct conceptual units.
Please split it into a JSON object with exactly two keys:
  "units": an array of strings, where each string is one complete conceptual unit.
  "remainder": a string containing any leftover text that does not form a complete unit.
If there is no leftover text, "remainder" should be an empty string.
Ensure your output is complete, valid JSON (with balanced braces) and contains no markdown formatting or extra text.
Return only the JSON.

Segment:
{segment}
"""
    messages = [
        {"role": "system", "content": "You are an expert at splitting text into distinct conceptual units. Your output must be complete valid JSON with two keys: 'units' and 'remainder'."},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(API_MAX_RETRIES):
        content = call_openai_with_retries(model="gpt-4", messages=messages, max_retries=1, max_tokens=500)
        if not content:
            print(f"Error: Received no response on split attempt {attempt+1}.")
            continue
        content = clean_code_block(content)
        try:
            result_obj = json.loads(content)
            if isinstance(result_obj, dict) and "units" in result_obj and "remainder" in result_obj:
                return result_obj
        except Exception as e:
            print(f"Error parsing split response on attempt {attempt+1}: {e}")
            print("Response content:", content)
            result_obj = fix_truncated_json(content)
            if result_obj is not None and isinstance(result_obj, dict) and "units" in result_obj and "remainder" in result_obj:
                return result_obj
            result_obj = attempt_fix_json(content)
            if result_obj is not None and isinstance(result_obj, dict) and "units" in result_obj and "remainder" in result_obj:
                return result_obj
        print(f"Retrying split_multiple_concepts (attempt {attempt+2})...")
    # Fallback: return the whole segment as one unit
    return {"units": [segment], "remainder": ""}

def summarize_concept(text, parent_tags=""):
    """
    Summarize a conceptual unit into a coherent summary of at most 1000 characters.
    Parent tags (if provided) will be prepended for context.
    """
    prompt = f"""Summarize the following conceptual unit into a coherent summary of at most 1000 characters.
If parent tags are provided, prepend them as context.
Do not change the core meaning.

Parent tags: {parent_tags}
Text:
{text}

Summary:"""
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": prompt}
    ]
    summary = call_openai_with_retries(model="gpt-4", messages=messages, max_retries=API_MAX_RETRIES, max_tokens=1024)
    if summary:
        return summary.strip()
    # Fallback: truncate manually if API call fails
    return (parent_tags + " " + text)[:1000]

# ---------------------------
# Semantic Segmentation of Document
# ---------------------------
def process_document_semantic(document_text, page_size=1000, target_size=1000):
    """
    Pages through the document in increments of page_size.
    Accumulates text until the LLM indicates that the accumulated text
    is a complete conceptual unit, contains multiple concepts, or is still partial.
      - If the segment is "complete", it is committed as one knowledge unit.
      - If it is "multiple", we split it using the LLM and carry forward any remainder.
      - If it is "partial", we accumulate further text.
    Finally, any knowledge unit longer than target_size is summarized.
    Returns a flat list of knowledge units.
    """
    pos = 0
    current_segment = ""
    knowledge_units = []

    while pos < len(document_text):
        next_page = document_text[pos: pos + page_size]
        pos += page_size
        current_segment += next_page

        status = query_concept_status(current_segment)
        print(f"Segment status: {status} (accumulated {len(current_segment)} chars)")
        if status == "complete":
            knowledge_units.append(current_segment.strip())
            current_segment = ""
        elif status == "multiple":
            result_obj = split_multiple_concepts(current_segment)
            units = result_obj.get("units", [])
            remainder = result_obj.get("remainder", "")
            knowledge_units.extend([unit.strip() for unit in units])
            current_segment = remainder  # carry forward remainder
        elif status == "partial":
            continue
        else:
            knowledge_units.append(current_segment.strip())
            current_segment = ""

    if current_segment.strip():
        status = query_concept_status(current_segment)
        if status == "multiple":
            result_obj = split_multiple_concepts(current_segment)
            units = result_obj.get("units", [])
            knowledge_units.extend([unit.strip() for unit in units])
        else:
            knowledge_units.append(current_segment.strip())

    final_units = []
    for unit in knowledge_units:
        if len(unit) > target_size:
            print(f"Summarizing unit of length {len(unit)}")
            summary = summarize_concept(unit, parent_tags="")
            final_units.append(summary)
        else:
            final_units.append(unit)

    return final_units

# ---------------------------
# Document Processing and File Handling
# ---------------------------
def is_supported_file(file_path):
    """Check if a file is a supported type (PDF or plain text)."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type in ["application/pdf", "text/plain"]

def process_document(file_path):
    """
    Reads a document from file_path (using pdfminer for PDFs or UTF-8 for text)
    and returns a dict with the full text and the semantic knowledge units.
    Unsupported file types are skipped.
    """
    print(f"\nProcessing file: {file_path}")
    
    if not is_supported_file(file_path):
        print(f"Skipping unsupported file type: {file_path}")
        return {"document": "", "chunks": []}
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            content = extract_text(file_path)
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {e}")
            content = ""
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            content = ""
    if not content.strip():
        print(f"Warning: {file_path} is empty or could not be read.")
    print("Segmenting document semantically...")
    units = process_document_semantic(content)
    return {"document": content, "chunks": units}

def find_and_process_files(dir_path):
    """
    Recursively processes all files in dir_path.
    Returns a dict with:
      - "documents": list of full document texts.
      - "chunks": flat list of all semantic knowledge units.
    """
    documents = []
    chunks = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_full_path = os.path.join(root, file)
            result = process_document(file_full_path)
            documents.append(result["document"])
            chunks.extend(result["chunks"])
    return {"documents": documents, "chunks": chunks}

def get_unique_filename(base_name="knowledge.json"):
    """Generate a unique filename to avoid overwriting an existing file."""
    base, ext = os.path.splitext(base_name)
    counter = 1
    new_filename = base_name
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# ---------------------------
# Main Execution
# ---------------------------
def main():
    global API_MAX_RETRIES  # Allow CLI to override the default retry count
    parser = argparse.ArgumentParser(description="Extract structured knowledge from documents in a folder.")
    parser.add_argument("folder", help="Folder containing documents to process.")
    parser.add_argument("--output", default="knowledge.json", help="Output JSON file (default: knowledge.json).")
    parser.add_argument("--force", action="store_true", help="Overwrite output file without prompting.")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for OpenAI API failures (default: 3).")
    
    args = parser.parse_args()
    API_MAX_RETRIES = args.retries

    script_dir = Path(__file__).resolve().parent
    starting_path = script_dir / args.folder
    if not starting_path.exists():
        print(f"Error: Folder {starting_path} does not exist. Exiting.")
        sys.exit(1)
    
    env_path = script_dir / ".env"
    ensure_env_file(env_path)
    api_key = get_api_key(env_path)
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    print(f"\nProcessing files in: {starting_path}")
    result = find_and_process_files(str(starting_path))

    output_file = args.output
    if os.path.exists(output_file) and not args.force:
        output_file = get_unique_filename(output_file)
        print(f"Output file exists. Saving as: {output_file}")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nDone processing files. Data saved to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(0)
