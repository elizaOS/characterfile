#!/usr/bin/env python3
import sys
import json
import os
import argparse

# This version converts the .js version to .py, and adds some command line improvements.

def prompt_user(question, default_value=''):
    """Prompt the user for input with a default value."""
    print()  # Blank line for spacing
    answer = input(f"{question} [{default_value}]: ").strip()
    return answer if answer else default_value

def read_json_file(file_path, retries=0):
    """Read and parse a JSON file with retry logic for JSON errors."""
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)  # Exit immediately for missing files

    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB threshold
        print(f"Warning: {file_path} is large (>100MB). Processing may be slow.")

    attempt = 0
    while attempt <= retries:
        try:
            with open(file_path, "r", encoding="utf8", errors="replace") as f:
                content = f.read()
            return json.loads(content)  # Load JSON normally
        except json.JSONDecodeError as error:
            print(f"Error: Invalid JSON format in {file_path}: {error}")
            if attempt == retries:
                print("Max retries reached. Exiting.")
                sys.exit(1)
            print(f"Retrying... ({attempt + 1}/{retries})")
            attempt += 1

def write_json_file(file_path, data, force=False):
    """Write JSON data to a file, avoiding accidental overwrites."""
    if os.path.exists(file_path) and not force:
        base, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        file_path = f"{base}_{counter}{ext}"
        print(f"Output file exists, saving as: {file_path}")

    try:
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully wrote JSON file: {file_path}")
    except Exception as error:
        print(f"Error writing JSON file {file_path}: {error}")
        sys.exit(1)

def merge_knowledge(character, knowledge):
    """Merge knowledge data into character JSON, preserving existing data."""
    if "knowledge" in character:
        if isinstance(character["knowledge"], dict) and isinstance(knowledge, dict):
            character["knowledge"].update(knowledge)
        elif isinstance(character["knowledge"], list) and isinstance(knowledge, list):
            character["knowledge"].extend(knowledge)
        else:
            print("Warning: Overwriting existing 'knowledge' field in character JSON.")
            character["knowledge"] = knowledge
    else:
        character["knowledge"] = knowledge
    return character

def main():
    """Main function to process JSON merging."""
    parser = argparse.ArgumentParser(description="Merge a character JSON with a knowledge JSON.")
    parser.add_argument("character", help="Path to the character JSON file")
    parser.add_argument("knowledge", help="Path to the knowledge JSON file")
    parser.add_argument("output", nargs="?", help="Path to the output JSON file (optional)")
    parser.add_argument("--retries", type=int, default=0, help="Number of times to retry on JSON errors")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file without prompting")

    args = parser.parse_args()

    try:
        character = read_json_file(args.character, retries=args.retries)
        knowledge = read_json_file(args.knowledge, retries=args.retries)

        if character is None or knowledge is None:
            print("Error: Failed to load one or both JSON files. Exiting.")
            sys.exit(1)

        # Determine output file name
        output_file_path = args.output
        if not output_file_path:
            character_name = character.get("name", "character").replace(" ", "_")
            output_file_path = f"{character_name}.knowledge.character.json"

        # Merge data
        updated_character = merge_knowledge(character, knowledge)

        # Write output file
        write_json_file(output_file_path, updated_character, force=args.force)

        print("Script execution completed successfully.")

    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(0)

if __name__ == '__main__':
    main()
