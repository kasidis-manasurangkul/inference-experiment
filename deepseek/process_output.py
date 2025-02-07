#!/usr/bin/env python3

import json
import re

def process_text(raw_text: str) -> str:
    """
    Convert literal '\n' into actual newlines and remove certain formatting markers.
    """
    # Replace literal "\n" with actual newline
    text = raw_text.replace("\\n", "\n")
    
    # Remove "**" markers used for bold
    text = text.replace("**", "")
    
    # Optionally remove other markup, e.g. triple backticks or underscores, if desired:
    # text = text.replace("```", "")
    # text = text.replace("*", "")
    # text = text.replace("_", "")
    
    # If you want to remove leftover multiple blank lines, you could do a small cleanup:
    # text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # merges triple+ blank lines into double
    return text

def main():
    input_json = "result/synthesis_data_validated.json"     # <-- Your JSON input file here
    output_txt = "result/output_conversations.txt"
    
    # Load the JSON
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_txt, "w", encoding="utf-8") as out:
        # Write an optional heading
        out.write("=== Processed Conversations ===\n\n")
        
        for idx, item in enumerate(data):
            # Some items might have "input" or "output" or both
            # Also includes "kb_name" and "index" if present
            
            kb_name = item.get("kb_name", "N/A")
            record_index = item.get("index", idx)
            
            # We process the "input" and "output" fields if they exist
            raw_in = item.get("input", "")
            raw_out = item.get("validated_output", "")
            
            # Transform them
            processed_input = process_text(raw_in)
            processed_output = process_text(raw_out)
            
            # Write to the .txt file in a more readable format
            out.write(f"--- Conversation Record #{record_index} (KB={kb_name}) ---\n")
            out.write("[Output]:\n")
            out.write(processed_output + "\n\n")
            
            # Optionally, add a separator line
            out.write("="*70 + "\n\n")

    print(f"Done. Processed text saved to '{output_txt}'.")

if __name__ == "__main__":
    main()
