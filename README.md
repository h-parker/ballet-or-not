# It's Not Rockette Science: Determining if a Classical Piece is from a Ballet
I had the idea to classify classical pieces as ballet or non-ballet pieces a few weeks ago, right before I saw Balanchinne's _Jewels_. _Jewels_ isn't a typical ballet -- it is comprised of three disjoint acts (Emeralds, Rubies, and Diamonds) with no real story. Rather than tell a story, it an exposition of talent and style and it is evocative of the style (French, American, and Russian, respectively) the act represented. Because this ballet seemed to antithetical to a traditional ballet, it got me thinking: what is a ballet, anyway? How does a ballet sound? Is it just any classical music, pretty costumes and sets, and some pirouettes? In my heart, I thought _no, of course not!_, and so I set out to prove it (or at least part of it) by creating a classifier of ballet music. 

## Data
### Source
I began by creating two playlists on Spotify -- one playlist had about 1,000 ballet pieces, and the other had about 1,000 non-ballet, classical pieces. Then I used Spotify's API via the Spotipy library to get their features and analysis for each piece. Later I scraped Last.fm to get genres when trying to find patterns in misclassified data, but more on that later!

### Features
- Features - Spotify gives developers a few features that are ready to use from the get go.
  - Acousticness - A continous measure from 0 to 1; closer to 1 represents a higher confidence that the track is acoustic.
  - Danceability - A continuous measure of how suitable the song is to dancing; closer to 1 represents a higher level of danceability.
  - Energy - A continous measure of intensity and activity throughout the whole song; closer to 1 represents that the track is more intense.
  - Instrumentalness - A continous measure of confidence in whether the song has vocals or not; above .5 represents instrumental music, whereas closer to 0 represents music with with voices. ("oohs" and "aahs" are _not_ treated as vocals).
  - Key - They key the track is in, where 0 is C, 1 is C#, 2 is D, 3 is D#, 4 is E, 5 is F#, etc.
  - Liveness - A continous measure of confidence that the song is live; .8 or higher represents strong confidence that the song is live.
  - Loudness - The loudness in decibels (typically ranging between -60 and 0).
  - Mode -  A categorical representation of the modality of the track -- 0 (minor) or 1 (major).
  - Speechiness - A measure of words in a song -- .33 and below are music, between .33 and .66 are music with spoken words, like rap, and values above .66 are probably podcasts,  audiobooks, or the like.
  - Tempo - the pace of the song, in beats per minute.
  - Time signature - the meter of the song, measured in beats per bar/measure.
  - Valance -the emotion of the song -- closer to 0 is more negative, closer to 1 is more positive. 
  
- Analysis - a list of dictionaries for each tatum, beat, segment, bar, and section were provided, but I only ended up using the segments and sections, so I'll only discuss those below.
  - Sections - 
  - Segments - A list of dictionaries for each segment, which was a further division of sections. In each dictionary, there were NOT FINISHED
  

## EDA & Feature Engineering

## Modeling
### Choosing the Model
Logistic regression, KNN, Random Forest, SVM, and XGBoost were all considered, and performed very similarly. After parameter tuning through grid search, the top 3 contenders were Random Forest, SVM, and XGBoost. These the top three based on accuracy, AUC-ROC, and F1 score, since I wanted to capture the model to penalize false positives and false negatives roughly equally. In the end, I chose Random Forest because it was a good balance of accuracy and interpretability -- SVM and XGBoost only performed slightly better, while failing to be interpretable. 

### Interpretation of the Model's Feature Importance

### What the Model Got Wrong

### Making Predictions with the Model 

## Next Steps
