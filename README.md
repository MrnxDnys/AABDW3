# AABDW3
## Assignment 3: Correctly classify chat messages to corresponding Twitch channel in real-time using Spark

### 3.1 Problem Statement
The goal of the third assignment was to construct a model able to classify textual
chat data from a real-time incoming stream. The data was coming from
Twitch channels that were chosen by us. From these channels, first a set of historical
data needed to be collected that could then be used for training. Next,
a predictive model had to be fitted using this data set. And finally this model
needed to be deployed in order to classify real-time incoming chat messages to
their corresponding channel.

### 3.2 Methodology
For this task we used PySpark and the included MLlib. We worked mostly in
Jupyter notebooks and Python scripts. Code as well as the data was shared
using Git. If the amount of data had gotten any bigger, another solution would
have been necessary to share the data.
Data Gathering
The first part of the assignment consisted of gathering enough historical data
to fit a decent predictive model. For this an endpoint was provided to which we
could listen. To set up the connection a local proxy was run to which the desired
channels were presented as parameters. Once this proxy was up and running, a
streaming thread could be set up to listen to the endpoint and save all incoming
data in a local folder.
To make sure we had enough data and to be able to try out different combinations
and languages we focused on 6 different channels. In total we gathered
1.5 million messages as shown in figure 9. Since some streamers cover different
topics (multiple games and events), we gathered this data over several streaming
sessions for a duration of more than one month.
The data we collected from listening to the endpoint was saved in separate
folders for every 10 seconds of streaming. This amounted to a large folder containing
a huge amount of subfolders (+300.000). To be able to work with this
data we wrote a script, shown in listing 1.1 looping through the folders and
transferring all suitable data to one JSON file.
        
        1 def preprocessing ( cd_path , data_folder , data_all_path ):
        2 """ Load all raw data and compile it into a single json
        file for convenience . """
        3
        4 data_path = os. path . join ( cd_path , data_folder )
        5 data_raw = []
        6 for root_path , _, file_names in os. walk ( data_path ):
        7 for f in file_names :
        8 if f. startswith (" part "):
        9 file_path = os. path . join ( root_path , f)
        10 with open ( file_path ) as ff:
        11 for line in ff:
        12 data_raw . append ( json . loads ( line ))
        13 with open ( data_all_path , ‘w’) as f:
        14 json . dump ( data_raw , f)
        15
        16
        17 if __name__ == " __main__ ":
        18 cd_path = os. path . dirname (os. path . abspath ( __file__ ))
        19
        20 parser = argparse . ArgumentParser ()
        21 parser . add_argument ("- dp", "-- data_path ", help =" Specify
        path to data .",
        22 default =os. path . join ( cd_path , " data "))
        23 parser . add_argument ("- dn", "-- data_fname ", help =" Specify
        data json file name .",
        24 default =" data ")
        25 args = parser . parse_args ()
        26
        27 data_all_filename = args . data_fname + ". json "
        28 data_all_path = os. path . join ( cd_path , data_all_filename )
        29 if not os. path . exists ( data_all_path ):
        30 preprocessing ( cd_path , args . data_path , data_all_path )
        31 print (" Data file was compiled .")
        32 else :
        33 print (" Compiled data file with the same name already
        exists .")
Listing 1.1: The data extraction script.
Preprocessing

The chat messages on Twitch tend to be very chaotic, random and full of emotes
and slang as can be seen in figure 10. This nonsensical nature of the data made
22
preprocessing especially challenging. Text cleaning techniques that would yield
good result on normal textual data therefore did not perform as well on the chat
messages.
To featurize this data we set up a preprocessing pipeline. This way the same
preprocessing steps that have been performed on the training data can be performed
on incoming chat messages. The design of the pipeline is shown in listing

### 1.2. First the channel of the message is identified as the label to be predicted.
Then username and message are concatenated with a space in between. This was
done using a user defined SQL Transformer. Next all text is set to lowercase and
words are tokenized by splitting on whitespace. Then stop words are removed.
Finally, word frequencies are computed, which can then be used to fit a model.
The word frequency is the number of times a word occurs in a chat message
divided by the total number of words in that chat message. A chat message is
then characterized by the vector of its word frequencies in a space defined by
one dimension for each word in the vocabulary.
        
        1 # Encoding of labels
        2 encoder = StringIndexer ( inputCol =‘ channel ’, outputCol =‘ label ’
        )
        3 # Preprocess data by adding username to message
        4 sqlTrans = SQLTransformer ( statement =" SELECT *, CONCAT (
        username , ‘ ’, message ) AS allWords FROM __THIS__ ")
        5 # Extracting words form messages ( each message is vector of
        words )
        6 tokenizer = Tokenizer ( inputCol =‘ allWords ’, outputCol =‘ tokens ’
        )
        7 # Remove stopwords
        8 remover = StopWordsRemover ( inputCol =‘ tokens ’, outputCol =‘
        tokens_rmvd ’)
        9 # Using TF to featurize vector of words
        10 hashingTF = HashingTF ( inputCol =‘ tokens_rmvd ’, outputCol =‘
        features ’)
Listing 1.2: The preprocessing pipeline.

#### Model Fitting
For the classification we used Naive Bayes, a probabilistic classifier based on
Bayes theorem that is known to perform well on textual data. From the gathered
data two channels were selected to make it a binary classification task. Most
often these were the channels #asmongold and #hasanabi, since for these we
had the most data and they are both mainly in English. Next, if any of the two
channels had significantly more data, we corrected this by randomly sampling
(without replacement) until both channels were equally represented, even though
this might suggest that one channel simply had a more active chat. This set
was then split into a 80% training and 20% test set. Next, a grid-search tenfold
cross-validation was used to tune the smoothing parameter. We compared
several models’ performance on the test set and Naive Bayes came out on top
with a test set accuracy of 96.4%.
        
        1 # Apply Naive - Bayes
        2 nb = NaiveBayes ( smoothing =1.0 , modelType =" multinomial ")
        3 # Build a pipeline
        4 nbPipeline = Pipeline ( stages =[ sqlTrans , encoder , tokenizer ,
        remover , hashingTF , nb ])
Listing 1.3: The model and final pipeline.

#### Model Deployment
In order to classify chat messages from live streaming channels the model pipeline
had to be deployed. This way the same preprocessing steps are preformed on
real-time incoming data. The listing 1.4 shows the entire code snippet needed to
make prediction on live chat messages. Figure 11 shows some examples of the
output.
        
        1 globals ()[‘ models_loaded ’] = False
        2 globals ()[‘ my_model ’] = None
        3
        4 def process (time , rdd ):
        5 if rdd . isEmpty ():
        6 return
        7
        8 print (" ========= %s ========= " % str( time ))
        9
        10 # Convert to data frame
        11 df = spark . read . json ( rdd)
        12
        13 # Load in the model if not yet loaded :
        14 if not globals ()[‘ models_loaded ’]:
        15 # load in your models here
        16 cd_path = os. getcwd ()
        17 pipeline_folder = " pipeline_fitted "
        18 path_to_model = os. path . join ( cd_path , pipeline_folder
        )
        24
        19 globals ()[‘ my_model ’] = PipelineModel . load (
        path_to_model )
        20 globals ()[‘ models_loaded ’] = True
        21
        22 # And then predict using the loaded model :
        23 df_result = globals ()[‘ my_model ’]. transform (df)
        24 df_result . select (‘ channel ’,‘ message ’,‘label ’,‘ probability
        ’,‘ prediction ’). show ()
        25
        26 ssc = StreamingContext (sc , 10)
        27 lines = ssc. socketTextStream (" localhost ", 8080)
        28 lines . foreachRDD ( process )
        29 ssc_t = StreamingThread (ssc)
        30 ssc_t . start ()
Listing 1.4: Code snippet of script to deploy model and classify chat messages
coming in on a real-time data stream.

### 3.3 Discussion
During this assignment we came across multiple hurdles. Firstly the nature of the
data was challenging. We considered additional preprocessing steps like stemming
and n-gram, however since the chat messages are very different from normal
Advanced Analytics Assignment 25
text, we opted to not include this in our final pipeline. The most important preprocessing
step we performed turned out be including the username. Users can
subscribe to their favourite streamers on Twitch. These subscribers get certain
advantages like channel-specific emotes and are often very active in the chat.
The username can therefore be assumed to be a feature that is highly correlated
to the channel. We simply appended the username to the message before other
featurization steps. It is important to note that Twitch specific emotes and other
characters have been omitted even before we received the data from the provided
endpoint. This could have been an interesting feature to include but would have
needed extra preprocessing measures.
Since we gathered data from several different channels, we were able to
compare the performance of our model on different combinations of streamers.
Overall our model performed equally well when two English channels had
to be distinguished. As would be expected, two channels of different languages
were differentiated more easily.
For the featurization we experimented with computing the TF-IDF[10] values
for the chat messages. TF-IDF stands for “Term Frequency - Inverse Document
Frequency” and is a technique to vectorize textual data based on the frequency
of the words in one instance relative to all other instances. It is therefore an
elaboration of the normal bag-of-words technique. It can be computed as follows:
TF =
Number of times a given word X appears in the chat message
Number of words in the chat message
IDF = log (
Number of chat messages in the Corpus
Number of chat messages containing the word X
)
TFIDF = TF ∗ IDF
However this did not improve performance significantly on test data. And
even though we gathered quite a lot of data, since we are working on streaming
data it is hard to define a Corpus. On top of that the vocabulary used is highly
variable due to the slang and frequent misspellings. This might lead to words
occurring that have never occurred during training, which would then get abnormally
high significance. For this same reason we did not try word2vec[7], another
featurization technique using a neural net to vectorize words according to their
context. Finally we considered sentiment analysis. It would be interesting to see
how a package like VADER[4] would perform on this type of data. But since we
wanted a model that performs on multiple languages and we reached sufficient
performance we decided that this was out of scope.
For the model we also tried different classifiers. As a baseline we used logistic
regression and compared everything to its performance, which was of around 84%
accuracy on test data. Additionally we fitted K-means clustering, a random forest
and our final solution Naive Bayes. K-means clustering performed a lot worse
than expected. It was only able to get an accuracy of 61%. Maybe this is due to
the high dimensionality of the feature space. On top of that, K-means clustering
is very memory intensive when used in a deployed setting on live streaming
data. The random forest was also very memory intensive and performed only
a little better, but still a lot worse compared to the logistic regression. The
Naive Bayes was the only model to perform better than the baseline with an
average performance of 96%. Note that we only performed cross-validation on the
logistic regression and Naive Bayes. Another advantage of the Naive Bayes was
it’s short training time. This made it possible to evaluate different preprocessing
and featurization techniques.
Another challenge was that streamers do not necessarily cover the same topics
in different streaming episodes. This can result in a very different atmosphere in
the chat (e.g. excitement during a video game vs opinion on current events in
the news). Additionally, channels sometimes cover the same topics. During the
week that we were testing our model on live streaming data, our main English
channels were covering the judicial process of Amber Heard vs Johnny Depp.
Luckily we had a large data set covering a period of over one month including
many different streams of our channels. This made it possible for the model to
identify features that were highly channel-specific and that generalize well over
different topics.
However, in a deployed setting for a longer period of time it would be necessary
to take into account the concept drift of the chat messages. The topics
handled by one channel, but also the active users and even the slang and emotes
that these use, will be prone to changes over time. Therefore it is best to continue
gathering data for batch retraining even after the model has been deployed.
A real-life solution should also include monitoring of real-time performance to
assess the frequency of retraining. The timespan of the assignment was not sufficient
to cover this, so it has been left out of scope.

### 3.4 Conclusion
In this section of our report we discussed how we collected data from the chat of
active streaming channels on Twitch, used this data to classify the chat messages
according to their channels and deployed the resulting model to predict real-time
incoming chat messages. The main challenges were the nature of the data, being
textual and often nonsensical, and the fact that the model needed to be able
to predict in a deployed setting, which imposed limits on prediction time and
memory consumption. In the end we designed a pipeline for preprocessing and
prediction, achieving an accuracy of 96%.
