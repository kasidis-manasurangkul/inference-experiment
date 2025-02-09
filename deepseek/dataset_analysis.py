#!/usr/bin/env python3
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from rapidfuzz import fuzz

# Path to the combined JSON file
combined_file = "result/synthesized-conversation-v1-0.json"

if not os.path.exists(combined_file):
    print(f"{combined_file} does not exist.")
    exit(1)

# Load records from the combined JSON file
with open(combined_file, 'r', encoding='utf-8') as f:
    records = json.load(f)

def find_all_pair_similarities(entries):
    """
    Given a list of entries (dictionaries), compute fuzzy similarity for every unique pair.
    Returns a list of dictionaries, each containing:
      - kb_name, index1, index2, similarity, output1, output2.
    """
    pair_list = []
    n = len(entries)
    for i in range(n):
        text1 = entries[i].get("output", "")
        idx1 = entries[i].get("index")
        for j in range(i+1, n):
            text2 = entries[j].get("output", "")
            idx2 = entries[j].get("index")
            sim_score = fuzz.ratio(text1, text2)
            pair_list.append({
                "kb_name": entries[i].get("kb_name", "Unknown"),
                "index1": idx1,
                "index2": idx2,
                "similarity": sim_score,
                "output1": text1,
                "output2": text2
            })
    return pair_list

# --- Group records by kb_name ---
records_by_kb = defaultdict(list)
for record in records:
    kb = record.get("kb_name", "Unknown")
    records_by_kb[kb].append(record)

# --- Compute pairwise similarities per group and overall ---
all_pairs = []
avg_similarity_by_kb = {}  # kb_name -> average similarity for that kb

for kb, group in records_by_kb.items():
    if len(group) < 2:
        avg_similarity_by_kb[kb] = 0
        continue
    pairs = find_all_pair_similarities(group)
    all_pairs.extend(pairs)
    sim_values = [pair["similarity"] for pair in pairs]
    avg_similarity = sum(sim_values) / len(sim_values)
    avg_similarity_by_kb[kb] = avg_similarity

# --- Plot 2D Scatter Plot: KB groups sorted by average similarity ---
sorted_kb = sorted(avg_similarity_by_kb.items(), key=lambda x: x[1], reverse=True)
x_vals = list(range(len(sorted_kb)))
y_vals = [sim for kb, sim in sorted_kb]

plt.figure(figsize=(10, 6))
plt.scatter(x_vals, y_vals, color='blue')
plt.title("Average Similarity per KB Group (Sorted Descending)")
plt.xlabel("KB Group (ordered by average similarity)")
plt.ylabel("Average Similarity")
for i, (kb, sim) in enumerate(sorted_kb):
    plt.annotate(kb, (x_vals[i], y_vals[i]), fontsize=8, rotation=45)
plt.tight_layout()
scatter_plot_path = "result/average_similarity_scatter.png"
plt.savefig(scatter_plot_path)
plt.show()
print(f"Scatter plot saved to {scatter_plot_path}")

# --- Plot Line Graph: All pairwise similarities sorted from highest to lowest ---
# Sort pairs by similarity in descending order.
sorted_pairs = sorted(all_pairs, key=lambda x: x["similarity"], reverse=True)
# Use pair rank from 1 to len(sorted_pairs)
x_line = list(range(1, len(sorted_pairs) + 1))
y_line = [pair["similarity"] for pair in sorted_pairs]

plt.figure(figsize=(10, 6))
plt.plot(x_line, y_line, marker='o', linestyle='-', color='green')
plt.title("Pairwise Similarities Sorted (Highest to Lowest)")
plt.xlabel("Pair Rank")
plt.ylabel("Similarity")
plt.tight_layout()
line_plot_path = "result/pairwise_similarity_line.png"
plt.savefig(line_plot_path)
plt.show()
print(f"Line plot saved to {line_plot_path}")

# --- Save 100 Lowest Similarity Pairs to a Text File ---
sorted_pairs_low = sorted(all_pairs, key=lambda x: x["similarity"])
lowest_txt_output = "result/lowest_100_pairs.txt"
with open(lowest_txt_output, "w", encoding="utf-8") as f:
    f.write("100 Lowest Similarity Pairs:\n")
    f.write("=" * 80 + "\n")
    for pair in sorted_pairs_low[:100]:
        f.write(f"KB Name: {pair['kb_name']}\n")
        f.write(f"Record {pair['index1']} Output: {pair['output1']}\n")
        f.write(f"Record {pair['index2']} Output: {pair['output2']}\n")
        f.write(f"Similarity: {pair['similarity']:.2f}\n")
        f.write("-" * 80 + "\n")
print(f"100 lowest similarity pairs saved to {lowest_txt_output}")

# --- Save 100 Highest Similarity Pairs to a Text File ---
highest_txt_output = "result/highest_100_pairs.txt"
with open(highest_txt_output, "w", encoding="utf-8") as f:
    f.write("100 Highest Similarity Pairs:\n")
    f.write("=" * 80 + "\n")
    for pair in sorted_pairs[:100]:
        f.write(f"KB Name: {pair['kb_name']}\n")
        f.write(f"Record {pair['index1']} Output: {pair['output1']}\n")
        f.write(f"Record {pair['index2']} Output: {pair['output2']}\n")
        f.write(f"Similarity: {pair['similarity']:.2f}\n")
        f.write("-" * 80 + "\n")
print(f"100 highest similarity pairs saved to {highest_txt_output}")
