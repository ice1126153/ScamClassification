import praw
import json
from datetime import datetime


reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent=""
)

subreddit = reddit.subreddit("Scams")

output_file = "scams_posts.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for submission in subreddit.new(limit=20000): 
        
        record = {
            "date_utc": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),  
            "timestamp": submission.created_utc,  
            "title": submission.title,
            "text": submission.selftext,
            "subreddit": str(submission.subreddit),  
            "comments": submission.num_comments,  
            "url": submission.url  
        }
        
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f" {output_file}")