"""
DCS 625  NLP Analysis Script
Author: James Baah
Description:
    This script performs data cleaning and natural language processing (NLP)
    on a quotes dataset as part of Deliverable 3.
    It tests five hypotheses about the dataset using Python NLP techniques.
"""

# ==================== 1. IMPORT LIBRARIES ====================
import pandas as pd
from collections import Counter
from itertools import islice
import string
import re

# ==================== 2. LOAD ORIGINAL DATA ====================
# Load the raw quotes CSV (scraped from quotes.toscrape.com in Deliverable 2)
df = pd.read_csv("sample_data.csv")

# ======= 3. DATA CLEANING FUNCTION ==========
def clean_text(text):
    """
    Cleans a single quote by:
    - Handling missing values
    - Removing smart quotes and regular quotes
    - Replacing multiple spaces with a single space
    - Converting all text to lowercase
    - Removing punctuation
    - Stripping extra spaces from ends
    """
    if pd.isnull(text):
        return ""
    text = text.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# Apply cleaning to the 'quote' column
df["cleaned_quote"] = df["quote"].apply(clean_text)

# Save cleaned dataset to a new CSV
df.to_csv("cleaned_quotes.csv", index=False)

# ========== 4. HYPOTHESIS 1 ===============
# GOAL:
#   Identify the most frequently used positive/motivational words
#   in the dataset using a provided CSV file containing a list of positive words.
#
# WHY:
#   This aligns with the requirement to focus on uplifting language
#   instead of just general frequent words.
#
# INPUT:
#   - cleaned_quotes.csv (from the cleaning step above)
#   - positive_words.csv (a CSV with one column listing positive/motivational words)
#
# OUTPUT:
#   - Printed list of top 20 positive/motivational words with their frequencies.

# Load the positive words CSV
positive_words_df = pd.read_csv("positive_words.csv")

# Convert list of words to lowercase and store in a set for fast lookup
positive_words_set = set(positive_words_df["word"].str.lower())

# Tokenize all cleaned quotes into words
tokens = " ".join(df["cleaned_quote"]).split()

# Keep only tokens that are alphabetic AND in the positive words set
positive_tokens = [word for word in tokens if word.isalpha() and word in positive_words_set]

# Count the frequency of each positive word
positive_word_freq = Counter(positive_tokens)

# Display the top 20 most frequent positive/motivational words
print("\n=== Hypothesis 1: Top 20 Positive/Motivational Words ===")
for word, freq in positive_word_freq.most_common(20):
    print(f"{word}: {freq}")

# ============ 5. HYPOTHESIS 2 ====================
# GOAL:
#   Count second-person pronouns in quotes tagged as "inspirational".

second_person_words = {"you", "your", "yours", "yourself", "yourselves"}

# Filter for inspirational quotes using the 'tags' column
inspirational_df = df[df['tags'].fillna("").str.contains("inspirational", case=False)]

# Tokenize inspirational quotes
inspirational_tokens = " ".join(inspirational_df["cleaned_quote"]).split()

# Count second-person words
second_person_count = sum(1 for word in inspirational_tokens if word in second_person_words)

print("\n=== Hypothesis 2: Second-Person Word Usage in Inspirational Quotes ===")
print(f"Total inspirational quotes: {len(inspirational_df)}")
print(f"Second-person word count: {second_person_count}")
print(f"Total words in inspirational quotes: {len(inspirational_tokens)}")
print(f"Percentage: {(second_person_count / len(inspirational_tokens) * 100):.2f}%")

# ======== 6. HYPOTHESIS 3 ========
# GOAL:
#   Identify which authors use the most emotional words (positive & negative).

positive_emotions = {"love", "hope", "believe", "happy", "joy", "dream", "inspire", "grateful"}
negative_emotions = {"fear", "hate", "sad", "pain", "hurt", "fail", "alone", "lost"}

author_emotion_counts = {}

# Loop over each quote and count emotion words for the author
for _, row in df.iterrows():
    author = row["author"]
    words = row["cleaned_quote"].split()
    pos_count = sum(1 for word in words if word in positive_emotions)
    neg_count = sum(1 for word in words if word in negative_emotions)

    if author not in author_emotion_counts:
        author_emotion_counts[author] = {"positive": 0, "negative": 0}

    author_emotion_counts[author]["positive"] += pos_count
    author_emotion_counts[author]["negative"] += neg_count

# Sort authors by total emotional word usage
sorted_authors = sorted(author_emotion_counts.items(),
                        key=lambda x: sum(x[1].values()),
                        reverse=True)

print("\n=== Hypothesis 3: Top Authors by Emotional Language ===")
for author, counts in sorted_authors[:5]:
    print(f"{author}: Positive={counts['positive']}, Negative={counts['negative']}")

# ==================== 7. HYPOTHESIS 4 ====================
# GOAL:
#   Find the most common two-word (bigram) and three-word (trigram) phrases.

bigrams = list(zip(tokens, islice(tokens, 1, None)))
trigrams = list(zip(tokens, islice(tokens, 1, None), islice(tokens, 2, None)))

print("\n=== Hypothesis 4: Top 10 Bigrams ===")
for phrase, freq in Counter(bigrams).most_common(10):
    print(f"{' '.join(phrase)}: {freq}")

print("\n=== Hypothesis 4: Top 10 Trigrams ===")
for phrase, freq in Counter(trigrams).most_common(10):
    print(f"{' '.join(phrase)}: {freq}")

# ==================== 8. HYPOTHESIS 5 ====================
# GOAL:
#   Compare vocabulary diversity (unique/total ratio) among the top 5 authors.

# Get the top 5 most quoted authors
top_authors = df["author"].value_counts().nlargest(5).index.tolist()

# Calculate vocabulary diversity for each
author_vocab_stats = {}
for author in top_authors:
    author_quotes = df[df["author"] == author]["cleaned_quote"]
    words = " ".join(author_quotes).split()
    total_words = len(words)
    unique_words = len(set(words))
    diversity = unique_words / total_words if total_words > 0 else 0
    author_vocab_stats[author] = {
        "total_words": total_words,
        "unique_words": unique_words,
        "diversity": round(diversity, 3)
    }

print("\n=== Hypothesis 5: Vocabulary Diversity by Author ===")
for author, stats in author_vocab_stats.items():
    print(f"{author} -> Total Words: {stats['total_words']}, "
          f"Unique Words: {stats['unique_words']}, "
          f"Diversity Score: {stats['diversity']}")
