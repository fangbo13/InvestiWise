import praw
from transformers import pipeline

# 初始化BERT情感分析器
sentiment_analyzer = pipeline('sentiment-analysis')

# 分段处理长文本
def analyze_sentiment(text):
    max_length = 512
    sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    # 分段处理
    for i in range(0, len(text), max_length):
        segment = text[i:i+max_length]
        sentiment = sentiment_analyzer(segment)
        label = sentiment[0]['label']
        if label == 'POSITIVE':
            sentiments['POSITIVE'] += 1
        elif label == 'NEGATIVE':
            sentiments['NEGATIVE'] += 1
        else:
            sentiments['NEUTRAL'] += 1
    
    return sentiments

def get_reddit_sentiments(query):
    reddit = praw.Reddit(
        client_id='ByGHuaBLiK2AdpNTPWKlCA',  
        client_secret='KfB9LAgGXaJ7PhUzRFvNZr32P3g5lg',  
        user_agent='Haibo Fang'  
    )

    total_sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    total_results = 0
    processed_posts = 0
    reddit_posts = []

    for submission in reddit.subreddit('all').search(query, limit=100):
        title = submission.title
        selftext = submission.selftext
        
        # 合并标题和自文本进行情感分析
        text_to_analyze = f"{title}. {selftext}"
        sentiments = analyze_sentiment(text_to_analyze)
        
        # 累加每篇帖子的情感结果
        total_sentiments['POSITIVE'] += sentiments['POSITIVE']
        total_sentiments['NEGATIVE'] += sentiments['NEGATIVE']
        total_sentiments['NEUTRAL'] += sentiments['NEUTRAL']
        
        # 添加Reddit帖子的标题和URL到列表
        reddit_posts.append({
            'title': title,
            'url': submission.url
        })
        
        processed_posts += 1

    return {
        'total_results': processed_posts,
        'sentiments': total_sentiments,
        'reddit_posts': reddit_posts
    }
