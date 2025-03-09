#!/usr/bin/env python
"""
Merged Python file combining functionalities for:
 - Fetching Reddit posts using PRAW,
 - Calculating sentiment scores using FinBERT,
 - Interpolating sentiment time series, and
 - Processing tweet/text files from various folders and merging results.
"""

import os
import re
import time
import json
import glob
import logging
from datetime import datetime, timezone, timedelta

import praw
import pandas as pd
from tqdm import tqdm

# For sentiment analysis
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Setup Reddit API Instance
# -------------------------
reddit = praw.Reddit(
    user_agent=True,
    client_id="DSPYqJYKgnkPWC1hYN8Cnw",
    client_secret="2HlMZf-X1PnwsoyfsMvNvwvBGn6FrA",
    username="Lost-Recognition2843",
    password="Ihopeyouhaveagreatday1219"
)

# ---------------------------------------
# Function to Fetch Daily Posts from Reddit
# ---------------------------------------
def fetch_daily_posts_with_content_df(subreddits, keywords, start_date, end_date, limit=20):
    """
    Fetch exactly 'limit' posts per day per subreddit, restricted to those containing
    at least one keyword, and only collect comments containing the keywords as well.
    
    Parameters:
        subreddits : list of subreddit names (e.g. ["technology", "science"])
        keywords   : list of keywords to search (e.g. ["gemini", "google"])
        start_date : datetime object of the start date
        end_date   : datetime object of the end date
        limit      : number of posts to fetch per day, per subreddit

    Returns:
        pd.DataFrame with columns: ["date", "subreddit", "post", "main"]
    """
    # Build a single query string that uses OR to include any of the keywords
    search_query = " OR ".join(keywords)
    
    # Regex pattern (case-insensitive) to ensure text actually contains at least one keyword
    pattern = re.compile("|".join(keywords), re.IGNORECASE)

    data = []
    current_date = start_date

    while current_date <= end_date:
        date_start = int(current_date.timestamp())
        date_end = int((current_date + timedelta(days=1)).timestamp())

        for subreddit_name in subreddits:
            count = 0  # Track posts collected for the subreddit on this day
            subreddit = reddit.subreddit(subreddit_name)
            print(f"Searching r/{subreddit_name} on {current_date.strftime('%Y-%m-%d')}...")

            try:
                # Search for posts that contain any of our keywords
                for submission in subreddit.search(search_query, sort="relevant", time_filter="all"):
                    if date_start <= submission.created_utc < date_end:
                        if count >= limit:
                            break

                        # Double-check the submission actually contains one of our keywords
                        text_to_check = (submission.title or "") + "\n" + (submission.selftext or "")
                        if not pattern.search(text_to_check):
                            continue

                        # Capture the post content (or note if it's a link post)
                        post_content = submission.selftext if submission.selftext else "No content (link post)"

                        # Fetch top-level comments (limit to 10), strictly within the same day
                        submission.comments.replace_more(limit=0)
                        top_comments = [
                            comment
                            for comment in submission.comments.list()[:10]
                            if date_start <= comment.created_utc < date_end
                        ]

                        # Now only collect comments that also have at least one keyword
                        for comment in top_comments:
                            print(comment.body)
                            if pattern.search(comment.body):
                                comment_dt = datetime.utcfromtimestamp(comment.created_utc)
                                data.append({
                                    "date": comment_dt.strftime('%Y-%m-%d'),
                                    "subreddit": subreddit_name,
                                    "post": comment.body,
                                    "main": post_content
                                })

                        count += 1

            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {e}")

        current_date += timedelta(days=1)

    df = pd.DataFrame(data, columns=["date", "subreddit", "post", "main"])
    return df

# --------------------------------------
# Setup Sentiment Analysis (FinBERT)
# --------------------------------------
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = probabilities.cpu().numpy()[0]

    # Calculate score and normalize to [0,1]
    sentiment_score = probabilities[0] * 1 + probabilities[1] * -1
    normalized_score = (sentiment_score + 1) / 2
    return normalized_score

def process_dataframe(df, text_col='text', date_col='date', min_count=3):
    """
    Apply sentiment scoring and compute the average daily sentiment,
    skipping days with fewer than min_count texts.
    """
    df['sentiment_score'] = df[text_col].apply(get_sentiment_score)
    grouped = df.groupby(date_col)
    results = []
    for date_val, group_df in grouped:
        if len(group_df) < min_count:
            print(f"Skipping {date_val}: Only {len(group_df)} rows found")
            continue
        daily_score = group_df['sentiment_score'].mean()
        results.append({
            'Date': date_val,
            'Sentiment_Score': daily_score
        })
    result_df = pd.DataFrame(results)
    return result_df

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # === Step 1: Fetch Reddit Posts and Export to CSV ===
    # Google-related subreddits and keywords
    subreddits = ['google', "GoogleOne", "technology", "stocks", "investing",
                  "wallstreetbets", "ValueInvesting", "finance", "technology", "StocksAndTrading"]
    keywords = ["gemini", "google", "pixel", "gates", "bill", "GOOG", 'Alphabet', "Search", "AI"]
    start_date = datetime(2023, 6, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 2, 3, tzinfo=timezone.utc)
    daily_limit = 20
    results_df = fetch_daily_posts_with_content_df(subreddits, keywords, start_date, end_date, daily_limit)
    df_to_export = results_df[['date', 'subreddit', 'post']]
    df_to_export.to_csv("reddit_sentiment_data.csv", index=False)

    # CVS-related subreddits and keywords
    subreddits = ['cvs', "CVS_Workers", "pharmacy", "stocks", "investing",
                  "wallstreetbets", "ValueInvesting", "finance", "flu", "StocksAndTrading"]
    keywords = ["cvs", "pharmacy", "healthcare", "prescription", "wellness"]
    start_date = datetime(2023, 6, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 2, 3, tzinfo=timezone.utc)
    daily_limit = 20
    results_df = fetch_daily_posts_with_content_df(subreddits, keywords, start_date, end_date, daily_limit)
    df_to_export = results_df[['date', 'subreddit', 'post']]
    df_to_export.to_csv("reddit_sentiment_data_cvs.csv", index=False)

    # === Step 2: Sentiment Analysis and Interpolation on Reddit Data ===
    # Process Google-related Reddit data
    Google_df = pd.read_csv("reddit_sentiment_data.csv")
    daily_sentiment_df = process_dataframe(Google_df, text_col='post', date_col='date', min_count=1)
    daily_sentiment_df['Date'] = pd.to_datetime(daily_sentiment_df['Date'])
    df_interp = daily_sentiment_df.copy()
    df_interp.set_index('Date', inplace=True)
    full_date_range = pd.date_range(start='2023-06-01', end='2025-02-03', freq='D')
    df_interp = df_interp.reindex(full_date_range)
    df_interp.index.name = 'Date'
    df_interp['Sentiment_Score'] = df_interp['Sentiment_Score'].interpolate(method='time')
    df_interp['Sentiment_Score'] = df_interp['Sentiment_Score'].ffill().bfill()
    df_interp.reset_index(inplace=True)
    print(df_interp)
    df_interp.to_csv("Google_sentiment_data_with_interpolation.csv", index=False)

    # Process CVS-related Reddit data
    CVS_df = pd.read_csv("reddit_sentiment_data_cvs.csv")
    daily_sentiment_cvs_df = process_dataframe(CVS_df, text_col='post', date_col='date', min_count=1)
    daily_sentiment_cvs_df['Date'] = pd.to_datetime(daily_sentiment_cvs_df['Date'])
    df_interp_cvs = daily_sentiment_cvs_df.copy()
    df_interp_cvs.set_index('Date', inplace=True)
    full_date_range = pd.date_range(start='2023-06-01', end='2025-02-03', freq='D')
    df_interp_cvs = df_interp_cvs.reindex(full_date_range)
    df_interp_cvs.index.name = 'Date'
    df_interp_cvs['Sentiment_Score'] = df_interp_cvs['Sentiment_Score'].interpolate(method='time')
    df_interp_cvs['Sentiment_Score'] = df_interp_cvs['Sentiment_Score'].ffill().bfill()
    df_interp_cvs.reset_index(inplace=True)
    print(df_interp_cvs)
    df_interp_cvs.to_csv("CVS_sentiment_data_with_interpolation.csv", index=False)

    # === Step 3: Process JSON/Text Files from Folders and Compute Daily Sentiment ===
    # Helper function to process a folder with JSON lines files
    def process_folder(folder_path, output_csv, min_count=3):
        all_texts = []
        for filename in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing {folder_path}"):
            full_path = os.path.join(folder_path, filename)
            if os.path.isdir(full_path):
                continue
            date_str = filename  # Assuming filename represents the date
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data.get("text", "")
                    all_texts.append({
                        "text": text,
                        "date": date_str
                    })
        df_all = pd.DataFrame(all_texts)
        result_df = process_dataframe(df_all, text_col='text', date_col='date', min_count=min_count)
        result_df.to_csv(output_csv, index=False)
        print(f"Done! Results in {output_csv}")

    # Process folders (adjust folder names as needed)
    process_folder("AMZN", "amazon_daily_sentiment(using tweets).csv", min_count=3)
    process_folder("CVS", "cvs_daily_sentiment(using tweets).csv", min_count=3)
    process_folder("GOOG", "google_daily_sentiment(using tweets).csv", min_count=3)

    # === Step 4: Merge Sentiment Data CSVs ===
    # Merge Amazon data
    Amazon_df_1 = pd.read_csv("amazon_daily_sentiment(using tweets).csv")
    Amazon_df_2 = pd.read_csv("Google_sentiment_data_with_interpolation.csv")  # Assuming this is the intended file for Amazon
    merged_df = pd.concat([Amazon_df_1, Amazon_df_2], ignore_index=True)
    print(merged_df.head())
    merged_df.to_csv("Amazon_merged_file.csv", index=False)

    # Merge CVS data
    CVS_df_1 = pd.read_csv("cvs_daily_sentiment(using tweets).csv")
    CVS_df_2 = pd.read_csv("CVS_sentiment_data_with_interpolation.csv")
    merged_df = pd.concat([CVS_df_1, CVS_df_2], ignore_index=True)
    print(merged_df.head())
    merged_df.to_csv("CVS_merged_file.csv", index=False)

    # Merge Google data
    Google_df_1 = pd.read_csv("google_daily_sentiment(using tweets).csv")
    Google_df_2 = pd.read_csv("Google_sentiment_data_with_interpolation.csv")
    merged_df = pd.concat([Google_df_1, Google_df_2], ignore_index=True)
    print(merged_df.head())
    merged_df.to_csv("Google_merged_file.csv", index=False)
