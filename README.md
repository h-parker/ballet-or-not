# It's Not Rockette Science: Determining if a Classical Piece is from a Ballet
I had the idea to classify classical pieces as ballet or non-ballet pieces a few weeks ago, right before I saw Balanchinne's _Jewels_. _Jewels_ isn't a typical ballet -- it is comprised of three disjoint acts (Emeralds, Rubies, and Diamonds) with no real story. Rather than tell a story, it an exposition of talent and style and it is evocative of the style (French, American, and Russian, respectively) the act represented. Because this ballet seemed to antithetical to a traditional ballet, it got me thinking: what is a ballet, anyway? How does a ballet sound? Is it just any classical music, pretty costumes and sets, and some pirouettes? In my heart, I thought _no, of course not!_, and so I set out to prove it (or at least part of it) by creating a classifier of ballet music. 

- [Data](#Data)
  * [Source](#Source)
  * [Features](#Features)
- [EDA and Feature Engineering](#eda)
  * [Interesting Findings](#interestingfindings)
  * [Feature Engineering](#featureengineering)
- [Modeling](#modeling)
  * [Choosing the Model](#choosingthemodel)
  * [Interpretation of the Model's Feature Importance](#interpretation)
  * [What the Model Got Wrong](#whatitgotwrong)
  * [Making Predictions with the Model](#predictions)
- [Closing Thoughts](#closing)
- [Next Steps](#nextsteps)

## Data <a name="Data"></a>
### Source <a name="Source"></a>
I began by creating two playlists on Spotify -- one playlist had about 1,000 ballet pieces, and the other had about 1,000 non-ballet, classical pieces. Then I used Spotify's API via the Spotipy library to get their features and analysis for each piece. Later I scraped Last.fm to get genres when trying to find patterns in misclassified data, but more on that later!

### Features <a name="Features"></a>
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
  - Valance - the emotion of the song -- closer to 0 is more negative, closer to 1 is more positive. 
  
- Analysis - a list of dictionaries for each tatum, beat, segment, bar, and section were provided, but I only ended up using the segments and sections, so I'll only discuss those below.
  - Sections - Splits the song into chunks that sound distinctly different from the last section. The data I used (there were additional features I did not use) that was given with each section were:
    - Duration
    - Loudness
    - Tempo
    - Key
    - Mode 
    - Time signature
    These have the same definitions as those listed above, just specific to the section, rather than the whole song. 
  - Segments - A list of dictionaries for each segment, which was a further division of sections. Each segment sounds roughly the same. The data I used (again, there were additional features I did not use) that was given with each segment were:
    - Duration - same definition as above
    - Pitches - a column vector for which each position in the vector represents one of the 12 pitches (C, C#, D, etc.). Each value is between 0 and 1, representing how common the pitch was in the segment.
    - Timbre - A vector representing the quality of a music note, which, when expressed as a linear combination, represent the overall timbre of the segment. 
  

## EDA & Feature Engineering <a name="eda"></a>
### Interesting Findings <a name="interestingfindings"></a>
When initially exploring the data, I really wanted danceability to matter, even though I knew Spotify's measure is probably more suited to disco and salsa danceability. Nevertheless, I plotted it, and thought it might be interesting to others. Below is the graph of the distributions of danceability for ballet pieces and non-ballet pieces. You can see that the distributions are _slightly_ different, with the distribution of ballet pieces shifter more to the right, towards the 'more danceable' end of the spectrum, but not quite enough to emphatically declare that there is a significant difference.  

![danceability distribution](https://github.com/h-parker/ballet-or-not/blob/master/danceability_dist.png "danceability distribution")

Then, while looking at other distributions, I came across a feature with a difference in distribution that felt exciting -- valence! As you can see, there is a significant different in both the shape and center of the distributions of valance for ballet and non-ballet pieces. While both are right-skewed, the tail of non-ballet songs is much skinnier, and the peak is much taller. Thus, we conclude that non-ballet songs tend to evoke more negative emotions, whereas the emotion of ballet songs is more evenly spread (even though there are still more negatively evocotive ballet songs). 

![valance distribution](https://github.com/h-parker/ballet-or-not/blob/master/valance_dist.png "valance distribution")

### Feature Engineering <a name="featureengineering"></a>
I introduced a number of different aggregations of the data on the sections and segments, as the ways of parsing apart and boiling down the songs using this data were endless. I created the following new features:
- Using section data:
  - duration range
  - loudness range
  - key range
  - tempo range
  - mode range (equivalent to seeing if any section was in major)
  - time signature range
- Using segment data:
  - duration range
  - number of unique pitches (unique "strengths", you could say? since each is a measure how of strong the presence of that pitch was in the segment)
  - number of unique timbre values
  - mean pitch
In all honesty, I had ideas about what some of these features could mean in the context of ballet vs not a ballet, but for others, I figured "hmm, the range of the duration could be useful, since one would expect ballets to have shorter sections as the the story moves quickly and action is unfolding", thought "why not see if the key range is any different between the two!"

However, later I did think hard about what these features told me, and if I could do it over again, I would be more thoughtful. This is especially true if I had a lot more data, which would mean that including all of those features could really slow down the time it takes to run all those models. 

## Modeling <a name="modeling"></a>
### Choosing the Model <a name="choosingthemodel"></a>
Logistic regression, KNN, Random Forest, SVM, and XGBoost were all considered, and performed very similarly. After parameter tuning through grid search, the top 3 contenders were Random Forest, SVM, and XGBoost. These the top three based on accuracy, AUC-ROC, and F1 score, since I wanted to capture the model to penalize false positives and false negatives roughly equally. In the end, I chose Random Forest because it was a good balance of accuracy and interpretability -- SVM and XGBoost only performed slightly better, while failing to be interpretable. The Random Forest ended up with 82% accuracy, with an F1 score of 82% as well, which were both only 1% lower than SVM and XGBoost's respective metrics. The confusion matrix and ROC curve are shown below. 

![confusion matrix](https://github.com/h-parker/ballet-or-not/blob/master/rf_cm.png)
![roc curve](https://github.com/h-parker/ballet-or-not/blob/master/rf_roccurve.png)

### Interpretation of the Model's Feature Importance <a name="interpretation"></a>
When looking at the bar chart of feature importance, below, we can see that the top 5 most important features are:
1. Duration
2. Number of sections
3. Number of segments
4. Acousticness
5. Mean pitch

![feature importance barchart](https://github.com/h-parker/ballet-or-not/blob/master/rf_feature_importance.png "feature importance")

When I see this, it appears to me that the model captures the story element of ballet pieces -- the number of sections and segments increase with ballet pieces, as the story progresses, since you're constantly shifting perspective from one character to the next, watching plot twists unfold, seeing the introduction of new characters, and these are all revealed both visually and musically. Each of these plot points sound distinctly different, and they need to -- as an audience, we wouldn't understand what was going on if they didn't! My interpretation of the mean pitch also supports my idea of story, since I would imagine that the mean of the pitches would be higher in songs with obvious stories, since a story is often told through a range of pitches. Thus, the mean would be brought up by a higher value in each position in the column vector. 

### What the Model Got Wrong <a name="whatitgotwrong"></a>
In an effort to improve my model, I wanted to see what it was getting wrong. Since Spotify doesn't have genres associated with their songs (only with artists, which wouldn't be helpful, since an artist like "Evergreen Symphony" may play all sorts of classical music, and I wanted just the subgenre of a particular song), I webscraped Last.fm to get the user generated tags for each piece that was misclassified. Then, I plotted the frequency of each tag.

![misclassified tags](https://github.com/h-parker/ballet-or-not/blob/master/top_misclassified_genres.png "bar chart of frequency of tags of misclassified songs")

Above, we can see that the top 10 genres (ignoring tags like "composer" that are not indicative of the subgenre) are
- Romantic
- Russian
- Contemporary
- Baraoque
- 20th Century
- German
- Italian 
- Instrumental
- Opera
- Avante-Garde

I would've expected the classifier to correctly classify more unique-sounding music, such as Baroque, since Baroque music is really not 'balletic' (which I know from my own domain knowledge). So, I looked at the breakdown of misclassified data by ballet/non-ballet:

![misclassified non-ballets](https://github.com/h-parker/ballet-or-not/blob/master/top_nb_misclassified_genres.png "distribution of tags of misclassified non-ballets")
![misclassified ballets](https://github.com/h-parker/ballet-or-not/blob/master/top_b_misclassified_genres.png "distribution of tags of misclassified ballets")

However, even still, this wasn't particularly illuminating! It still seems that it slipped up on somewhat obvious songs! I expected orchestral pieces to be the number one genre misclassified as a ballet. I'm thinking that maybe it could be more illuminating if I maintained the groupings (since songs has multiple tags, and maybe they say more together), or if I found another source? An exploration for another day!


### Making Predictions with the Model <a name="predictions"></a>
The fun part! I tested a few songs:
- [A ballet piece (Sleeping Beauty Intro)](https://open.spotify.com/track/59pxmlKl59Fr5uU8CVKft3?si=73EE90IsTKy_TUx0XCxqYQ) - classified as a ballet
- [A classical piece (River Free)](https://open.spotify.com/track/1ANLBj90qVTEUmr4ClP9OL) - classified as a classical piece
- [A 'ballet sounding' classical piece (Elgar: Variations on an Original Theme, Op. 36, Enigma IX)](https://open.spotify.com/track/23ryVoyGTrfO3F0GblIfnz?si=Y7W8AdgKSu6ix_GhcUJFxA) - classified as a ballet

Then, for fun, just to see what would happen:
- [A Beatles song - "Across the Universe"](https://open.spotify.com/track/4dkoqJrP0L8FXftrMZongF?si=t1BESPPNSdihVLJXkCKXSQ); classified as a ballet
- [A Rolling Stones song - "Wild Horses"](https://open.spotify.com/track/52dm9op3rbfAkc1LGXgipW?si=CBBlxo57QAKgvw60-V1K7Q); classified as not a ballet
- [An Allman Brothers song (rock) - "Whipping Post"](https://open.spotify.com/track/5JBUpI6OGZahUqchMKe6UY?si=c1oXpWFcRcOkN44jjvW8Tg); classified as not a ballet


## Closing Thoughts <a name="closing"></a>
Though my classifier was not completely accurate, I'm not too upset! Through playing around with making predictions, I came to discover that it would misclassify songs that I, myself, misclassified. Is that really a misclassification? If you're looking to always accurately classify whether a song has been used in a ballet, then yes. However, if you're looking to accurately classify songs that are ballet songs or _could_ be ballet songs, then no! Personally, I'm more interested in the latter. I feel as though songs (like the third tested in the Making Predictions section) that sound balletic _should_ be classified as such, and in the end, I'm happy that my classifier was able to "hear" those distinctions. 


## Next Steps <a name="nextsteps"></a>
- Examine the distribution of tags across all pieces and incorporate groups of tags (since each song has multiple tags) to hopefully make the charts of what the model got wrong more meaningful
- Explore more deeply the distributions of the features that were the most importance and try and get a better grasp on why they matter so much and what they might tell us about the characteristics ballet pieces
  - I feel the worst about not spending more time on this! With only a week, I got so focused on creating a strong classifier that I lost sight of my focus -- I definitely will remember this trap in my next project.
 - Inspect the features that seem most important to PCA and feature regularization (for logistic regression)
  - I didn't get to spend nearly as much time as I would've liked on this, which I regret for the same reasons as above. 
