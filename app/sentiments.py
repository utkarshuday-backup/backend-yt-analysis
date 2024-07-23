from langdetect import detect, DetectorFactory
from textblob import TextBlob
import pandas as pd
import numpy as np
import demoji
import re

DetectorFactory.seed = 0


# def replace_emojis(text):
#   return demoji.replace_with_desc(text, sep=" ")


# def remove_colons_and_numbers(text):
#   pattern = r'[:\d]'
#   return re.sub(pattern, '', text)


# def is_english(text):
#   try:
#     if detect(text) == 'en':
#       return text
#     return np.nan
#   except:
#     return np.nan


# def clean_comments(comments):
#   df = pd.DataFrame(comments).drop_duplicates()
#   df[0] = df[0].apply(replace_emojis).apply(
#       remove_colons_and_numbers).apply(is_english)
#   df = df.dropna()
#   comments = df[0].tolist()
#   return comments


def replace_emojis(text):
  return demoji.replace_with_desc(text, sep=" ")


def remove_colons_and_numbers(text):
  pattern = r'[:\d]'
  return re.sub(pattern, '', text)


def is_english(text):
  try:
    if detect(text) == 'en':
      return text
    return np.nan
  except:
    return np.nan


def clean_comments(comments):
  # Convert the list of comments to a pandas Series
  comments_series = pd.Series(comments).drop_duplicates()

  # Apply the cleaning functions using vectorized operations
  comments_series = comments_series.apply(replace_emojis)
  comments_series = comments_series.apply(remove_colons_and_numbers)
  comments_series = comments_series.apply(is_english)

  # Drop any NaN values
  comments_series = comments_series.dropna()

  # Convert the cleaned Series back to a list
  cleaned_comments = comments_series.tolist()

  return cleaned_comments


sentiment_ranges = {
    'veryPositive': (0.6, 1.01),
    'postive': (0.2, 0.6),
    'neutral': (-0.2, 0.2),
    'negative': (-0.6, -0.2),
    'veryNegative': (-1.0, -0.6)
}


def getSentimentScores(comments, videoId):

  comments = clean_comments(comments)
  print(f'Cleaned commments for videoId {videoId} ...')

  sentiment_counts = {
      'veryPositive': 0,
      'postive': 0,
      'neutral': 0,
      'negative': 0,
      'veryNegative': 0
  }
  print(f'Analyzing commments for videoId {videoId} ...')
  i = 0
  for comment in comments:
    blob = TextBlob(comment)
    sentiment_score = blob.sentiment.polarity
    i += 1
    if i % 100 == 0:
      print(f'{i} comments analyzed for videoID {videoId}')
    for sentiment, (lower, upper) in sentiment_ranges.items():
      if lower <= sentiment_score < upper:
        sentiment_counts[sentiment] += 1
        break
  n = len(comments)
  positive = ((sentiment_counts['veryPositive'] +
               sentiment_counts['postive'])/n)*100
  negative = ((sentiment_counts['negative'] +
               sentiment_counts['veryNegative'])/n)*100
  neutral = ((sentiment_counts['neutral']/n)*100)
  result = dict(positivePercentage=positive,
                neutralPercentage=neutral,
                negativePercentage=negative,
                numbersAnalyzed=n,
                sentimentCounts=sentiment_counts
                )
  return result
