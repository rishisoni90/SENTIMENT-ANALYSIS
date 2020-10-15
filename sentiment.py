import pandas as pd
import tweepy
import matplotlib.pyplot as plt
import re
from googletrans import Translator
from textblob import TextBlob
from wordcloud import WordCloud


# obtain these codes by register to the Twitter Developer platform
consumer_key= 'CONSUMER KEY'
consumer_secret= 'CONSUMER SECRET'

access_token= 'ACCESS TOKEN'
access_token_secret= 'ACCESS TOKEN SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


# scrap tweets (w/o retweets) from specific user
# iterate and gathering up to 100 tweets (without retweets)
# lowercase function is helpful for upcoming data cleaning
user_tweet = api.user_timeline(screen_name = 'USER', count = 100, include_rts = False)
for tweet in user_tweet:
    print(tweet.text.lower())

df = pd.DataFrame([tweet.text.lower() for tweet in user_tweet], columns = ['tweets'])

# define function which helps us to clean our tweets
# powered by RegEx
def cleaningTweets(twt):
    twt = re.sub('@[A-Za-z0-9]+', '', twt)
    twt = re.sub('#', '', twt)
    twt = re.sub('https?:\/\/\S+', '', twt)

    return twt

# apply previous function to the current df
df.tweets = df.tweets.apply(cleaningTweets)

# stopwords are beneficial to remove unwanted words from our df
# stopwords in Indonesian
with open('stopwords.txt','r') as f:
    stop_word = [word.strip() for word in f]

# whole_words will contain all words from respective tweets
# and word_cloud do the job to visualize it
whole_words = " ".join([tweets for tweets in df.tweets])
word_cloud = WordCloud(width = 700, height = 500, random_state = 1,
                       min_font_size = 10, stopwords = stop_word).generate(whole_words)

# generate wordcloud in png file
word_cloud.to_file('WordCloud.png')


### SENTIMENT ANALYSIS USING TEXTBLOB ###


# due to TextBlob inaccuracy and incompability for Indonesian language
# first we'll have to translate from Indonesian to English
# iterate through every row in df, translate to english, and make a column for it
translator = Translator()
for index, row in df.iterrows():
        id_twt = df.iloc[index]['tweets']
        translation = TextBlob(id_twt)
        en_twt = translator.translate(translation,dest = 'en')
        df.at[index, str('english_tweets')] = str(en_twt.text)

# remove 'tweets' column that contains Indonesian tweets
df.drop(columns = ['tweets'], axis = 1, inplace = True)

# sentiment_polar: indicates emotions expressed in a tweet (positive, neutral, negative)
# sentiment_subject: indicates expression of personal feelings (positive, neutral, negative)
# sentiment_sort: sorting the polarity by their own categorical score
df['sentiment_polar'] = df['english_tweets'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
df['sentiment_subject'] = df['english_tweets'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
sentimentSort = ['Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral' for score in df.sentiment_polar]
df['sentiment_category'] = sentimentSort

# the sum count of three polarity category in dataframe form
polar_counts = df.sentiment_category.value_counts().rename_axis('category').reset_index(name = 'counts')

# sentiment Polarity Visualization using Matplotlib
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (10,10))
ax.pie(polar_counts['counts'], autopct ='% 1.1f %%', shadow = True, labels = polar_counts['category'],
      colors = ('gray','blue','red'), textprops = {'color':'black','fontsize':'20'})
ax.set_title('Sentiment Polarization')
ax.legend(polar_counts['category'], loc = 'upper right')

plt.show()
