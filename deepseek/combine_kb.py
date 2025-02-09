#!/usr/bin/env python3
import json
import os
import csv
import re

def extract_kb_number(kb_name):
    """
    Extracts the first integer found in the kb_name string.
    If no digits are found, returns 0.
    """
    match = re.search(r'\d+', kb_name)
    return int(match.group()) if match else 0

def contains_long_english_sequence(text, threshold=10):
    """
    Checks if the text contains more than `threshold` consecutive English words.
    The regex matches sequences of at least (threshold+1) words (case-insensitive).
    """
    pattern = re.compile(r'(?i)\b(?:[a-z]+(?:\s+|$)){' + str(threshold+1) + r'}')
    return bool(pattern.search(text))

def contains_only_allowed_characters(text):
    """
    Returns True if the text contains only allowed characters.
    Allowed characters include:
      - Thai characters (Unicode range: U+0E00–U+0E7F)
      - Basic ASCII printable characters (U+0020–U+007E)
      - Common additional punctuation found in text:
            “ (U+201C), ” (U+201D), ‘ (U+2018), ’ (U+2019),
            … (U+2026), – (U+2013), — (U+2014)
      - Whitespace (spaces, tabs, newlines, etc.)
    Otherwise, returns False.
    """
    # Build a character class that includes:
    #   - Thai: \u0E00-\u0E7F
    #   - Basic ASCII: \u0020-\u007E
    #   - Additional punctuation: “ ” ‘ ’ … – —
    allowed_pattern = re.compile(
        r'^[\u0E00-\u0E7F\u0020-\u007E\u201C\u201D\u2018\u2019\u2026\u2013\u2014\s]+$',
        re.UNICODE
    )
    return bool(allowed_pattern.match(text))

def main():
    # Hard-coded list of input JSON files to combine
    input_files = [
        "result/0-100kb.json",
        "result/101-200kb.json",
        "result/201-300kb.json",
        "result/301-400kb.json",
        "result/401-468kb.json"
    ]
    # Output files will be placed in the "result" folder
    json_output_file = "result/synthesized-conversation-v1-0.json"
    csv_output_file = "result/synthesized-conversation-v1-0.csv"

    combined = []
    # Read each file and extend combined list
    for fname in input_files:
        if not os.path.exists(fname):
            print(f"Warning: {fname} does not exist. Skipping.")
            continue
        with open(fname, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue

            # If the file contains a list of records, extend; if a single dict, append.
            if isinstance(data, list):
                combined.extend(data)
            elif isinstance(data, dict):
                combined.append(data)
            else:
                print(f"Unrecognized JSON structure in file {fname}")

    if not combined:
        print("No records found. Exiting.")
        return

    # For each record, remove the current "output" key and rename "validated_output" to "output"
    for record in combined:
        if "output" in record:
            del record["output"]
        if "validated_output" in record:
            record["output"] = record["validated_output"]
            del record["validated_output"]

    # Filter out records whose output contains "<think>" or "</think>"
    initial_count = len(combined)
    combined = [
        record for record in combined 
        if "<think>" not in record.get("output", "") and "</think>" not in record.get("output", "")
    ]
    removed_think_count = initial_count - len(combined)
    print(f"Removed {removed_think_count} records due to containing '<think>' or '</think>' in output.")

    # Further filter out records with more than 10 consecutive English words in the output.
    before_english_filter = len(combined)
    combined = [record for record in combined if not contains_long_english_sequence(record.get("output", ""))]
    removed_english_count = before_english_filter - len(combined)
    print(f"Removed {removed_english_count} records due to containing more than 10 consecutive English words in output.")

    # Further filter: remove records whose output contains any disallowed characters.
    before_allowed_filter = len(combined)
    combined = [record for record in combined if contains_only_allowed_characters(record.get("output", ""))]
    removed_allowed_count = before_allowed_filter - len(combined)
    print(f"Removed {removed_allowed_count} records due to containing characters outside the allowed set.")

    # Sort the records by the numeric value extracted from kb_name (from smallest to largest)
    combined.sort(key=lambda r: extract_kb_number(r.get("kb_name", "")))

    # Reassign new sequential indices starting from 0
    for i, record in enumerate(combined):
        record["index"] = i

    # Write the combined records to the JSON output file
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Combined {len(combined)} records into {json_output_file}")

    # For CSV, gather all keys from the combined records so all data is listed.
    all_keys = set()
    for record in combined:
        all_keys.update(record.keys())
    fieldnames = sorted(all_keys)

    # Write the combined records to the CSV output file
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in combined:
            writer.writerow(record)
    print(f"Combined records also written to {csv_output_file}")

if __name__ == "__main__":
    main()
