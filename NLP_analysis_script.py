"""
DCS 625 - NLP Analysis Script
Author: James Baah

Purpose
-------
End-to-end text cleaning and NLP analysis on the quotes dataset scraped in Deliverable 2.
This script assumes the presence of `sample_data.csv` (created by the crawler.py),
generates `cleaned_quotes.csv`, and answers five hypothesis questions (H1–H5).

WHAT CHANGED 
------------------
• H1 now FILTERS tokens using an in-code set of motivational/positive words.
  - No external positive_words.csv is loaded.
  - The rest of the pipeline (cleaning + H2–H5) is unchanged.

Inputs
------
• sample_data.csv
    Required columns: 'quote' (text), 'author' (string), 'tags' (string/CSV list)

Outputs
-------
• cleaned_quotes.csv (adds 'cleaned_quote' column)
• Printed results to the console for H1–H5

Run
---
conda activate quotes_env
python NLP_analysis_script.py
"""

import re
import string
from collections import Counter
from itertools import islice

import pandas as pd


# ===============
# 0) LOAD DATA
# =================
# Load the raw scraped data. This file comes from your Deliverable 2 crawler.py
df = pd.read_csv("sample_data.csv")

# Defensive fills for expected columns (prevents .apply errors if nulls appear)
df["quote"] = df["quote"].fillna("")
df["author"] = df.get("author", "")
df["tags"] = df.get("tags", "")


# ========================
# 1) CLEANING UTILITIES
# ============================
def clean_text(text: str) -> str:
    """
    Normalize a single quote string:
      • Remove smart quotes and straight quotes
      • Collapse multiple whitespace to a single space
      • Lowercase
      • Remove punctuation
      • Strip leading/trailing spaces

    Returns a clean, analysis-ready string (lowercase words separated by spaces).
    """
    if pd.isnull(text):
        return ""

    # Normalize curly/smart quotes and stray quotes
    text = (
        text.replace("“", "")
            .replace("”", "")
            .replace("‘", "")
            .replace("‘", "")
            .replace('"', "")
            .replace("'", "")
    )

    # Collapse any multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)

    # Lowercase for consistent token matching
    text = text.lower()

    # Remove punctuation (keep only word chars and spaces)
    # Using translate is fast and safe here.
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text.strip()


# Apply cleaning to the quote text and persist for downstream steps/inspection
df["cleaned_quote"] = df["quote"].apply(clean_text)

# Save a cleaned copy for the deliverable artifact
df.to_csv("cleaned_quotes.csv", index=False)


# ==========================================
# 2) HYPOTHESIS 1 — Most frequent motivational words 
# =======================
"""
- Filter tokens to motivational/positive

Approach:
- Define a compact yet meaningful word set below.
- Tokenize the cleaned corpus and keep only tokens in that set.
- Count frequencies and report the top items.
"""
motivational_words = {
    "achieve", "believe", "brave", "commit", "courage", "create", "dedication",
    "dream", "effort", "faith", "focus", "goal", "grace", "gratitude", "grateful",
    "grow", "hope", "imagine", "inspire", "journey", "joy", "kindness", "learn",
    "love", "motivate", "passion", "persevere", "positive", "power", "progress",
    "resilience", "rise", "smile", "strength", "success", "trust", "will", "win"
}

# Tokenize full cleaned corpus (quotes are already normalized to lowercase).
all_tokens = " ".join(df["cleaned_quote"]).split()

# Keep ONLY motivational words
motivational_tokens = [t for t in all_tokens if t in motivational_words]

# Count and display the most common motivational words
h1_freq = Counter(motivational_tokens)
print("\n========================================================")
print("H1: Top Motivational Words (in-code filter)")
print("========================================================")
for word, count in h1_freq.most_common(20):
    print(f"{word}: {count}")


# ===============================
# 3) HYPOTHESIS 2 — Second-person words in inspirational quotes
# =================
"""
Question:
- Do quotes tagged as “inspirational” use second-person language more often?

Method:
- Filter rows where 'tags' contains 'inspirational'
- Count second-person tokens within those quotes
"""
second_person_words = {"you", "your", "yours", "yourself", "yourselves"}

# Robust tag matching: case-insensitive contains "inspirational"
inspirational_df = df[df["tags"].fillna("").str.contains("inspirational", case=False)]
inspirational_tokens = " ".join(inspirational_df["cleaned_quote"]).split()

second_person_count = sum(1 for w in inspirational_tokens if w in second_person_words)
total_insp_tokens = len(inspirational_tokens)
share = (second_person_count / total_insp_tokens) * 100 if total_insp_tokens else 0.0

print("\n========================================================")
print("H2: Second-person usage in 'inspirational' quotes")
print("========================================================")
print(f"Second-person count: {second_person_count} out of {total_insp_tokens} tokens "
      f"({share:.2f}% of inspirational tokens).")


# =============
# 4) HYPOTHESIS 3 — Emotional word usage by author
# ===================
"""
Question:
- Which authors most frequently use emotional language?
- Simple lexicon approach with small positive/negative sets.

Note:
- This is not full sentiment analysis; it’s a targeted, interpretable count.
"""
positive_emotions = {
    "love", "hope", "believe", "happy", "joy", "dream", "inspire", "grateful"
}
negative_emotions = {
    "fear", "hate", "sad", "pain", "hurt", "fail", "alone", "lost"
}

author_emotion_counts: dict[str, dict[str, int]] = {}

for _, row in df.iterrows():
    author = row["author"]
    words = row["cleaned_quote"].split()

    pos = sum(1 for w in words if w in positive_emotions)
    neg = sum(1 for w in words if w in negative_emotions)

    if author not in author_emotion_counts:
        author_emotion_counts[author] = {"positive": 0, "negative": 0}

    author_emotion_counts[author]["positive"] += pos
    author_emotion_counts[author]["negative"] += neg

# Rank authors by total emotional terms (pos + neg) and show top 5
ranked_authors = sorted(
    author_emotion_counts.items(),
    key=lambda kv: kv[1]["positive"] + kv[1]["negative"],
    reverse=True
)[:5]

print("\n========================================================")
print("H3: Top authors by emotional language (positive + negative)")
print("========================================================")
for author, counts in ranked_authors:
    print(f"{author}: {counts}")


# ==================
# 5) HYPOTHESIS 4 — Most common bigrams and trigrams
# ====================
"""
Question:
- What common 2- and 3-word phrases occur across the corpus?

Method:
- Construct bigrams/trigrams from the token stream and count top items.
"""
bigrams = list(zip(all_tokens, islice(all_tokens, 1, None)))
trigrams = list(zip(all_tokens, islice(all_tokens, 1, None), islice(all_tokens, 2, None)))

top_bigrams = Counter(bigrams).most_common(10)
top_trigrams = Counter(trigrams).most_common(10)

print("\n=====================")
print("H4: Top 10 bigrams")
print("======================")
for (w1, w2), c in top_bigrams:
    print(f"{w1} {w2}: {c}")

print("\n========================")
print("H4: Top 10 trigrams")
print("======================")
for (w1, w2, w3), c in top_trigrams:
    print(f"{w1} {w2} {w3}: {c}")


# =======================
# 6) HYPOTHESIS 5 — Vocabulary diversity by author (top 5 quoted)
# ============================
"""
Question:
- How does vocabulary variety differ across the most quoted authors?

Metric:
- Diversity = unique_tokens / total_tokens (per-author)
"""
top_authors = df["author"].value_counts().nlargest(5).index.tolist()

author_vocab_stats: dict[str, dict[str, float | int]] = {}
for author in top_authors:
    author_quotes = df.loc[df["author"] == author, "cleaned_quote"]
    words = " ".join(author_quotes).split()
    total = len(words)
    unique = len(set(words))
    diversity = (unique / total) if total else 0.0
    author_vocab_stats[author] = {
        "total_words": total,
        "unique_words": unique,
        "diversity": round(diversity, 3),
    }

print("\n========================================================")
print("H5: Vocabulary diversity by author (Top 5 most quoted)")
print("========================================================")
for a, stats in author_vocab_stats.items():
    print(f"{a}: {stats}")


# =============================================================================
# OPTIONAL: MAIN GUARD (kept simple since this file runs as a script)
# =============================================================================
if __name__ == "__main__":
    print("\nAnalysis complete. Cleaned data saved to 'cleaned_quotes.csv'.")
