
#Importing libraries
library("tidyverse")
library("tm")
library("SnowballC")
library("wordcloud2")
library("RColorBrewer")
library("ggplot2")
library("RWeka")
library("reshape2")


# PHASE 1: PREPROCESSING

# reading the dataset
reviews <- read.csv('../../UPM/1stYear/IntelligentSystems/NLP/NLPProject/2015GiftCard.csv', header = TRUE)
reviews <- reviews %>% select(reviewerID, reviewText)

# dataset transformation to be used in Corpus
reviews <- reviews %>% 
  rename(
    doc_id = reviewerID,
    text = reviewText
  )
# Number of reviews
count(reviews)

# creating a corpus
corp <- Corpus(DataframeSource(reviews))

#removing unimportant signs
clean <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(corp, clean, "!")
docs <- tm_map(corp, clean, ":")
docs <- tm_map(corp, clean, "?")
docs <- tm_map(corp, clean, "<")
docs <- tm_map(corp, clean, ">")
docs <- tm_map(corp, clean, "%")



# expanding stopwords
myStopwords = c(stopwords(),"gift", "card", "cards", "amazon","can", "will", "gift","s", "ve", "t")

# generating term document matrix
tdm = TermDocumentMatrix(corp,
                         control=list(stopwords = myStopwords,
                                      tolower = T,
                                      stopwords = T,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stripWhitespace = T))

m <- as.matrix(tdm)
vect <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(vect),freq=vect)
# returning top 10 words
head(d, 10)

# creating a first wordcloud for investigation
wordcloud2(data=d, size = 1, shape = 'pentagon', color=brewer.pal(10, "BrBG"))

# plotting a barchart of the 15 most popular words 
ggplot(head(d,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Word") + ylab("Frequency") +
  ggtitle("Most frequent words")

# PHASE 2: ASSOCIATION ANALYSIS

# searching for associations with one of the most popular words in reviews
findAssocs(tdm, terms= c("christmas"), corlimit=c(0.15))
findAssocs(tdm, terms= c("great"), corlimit=c(0.15))
findAssocs(tdm, terms= c("love"), corlimit=c(0.40))


# PHASE 3: N-GRAM ANALYSIS

#creating VCorpus for N-grams application
corp <- VCorpus(DataframeSource(reviews))
corp.ngrams = tm_map(corp,removeWords,c(myStopwords))

# BIGRAMS GENERATION
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram = TermDocumentMatrix(corp.ngrams,
                         control=list(stopwords = myStopwords,
                                      tolower = T,
                                      stopwords = T,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stripWhitespace = T,
                                      tokenize = BigramTokenizer))

freq = sort(rowSums(as.matrix(tdm.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
# returning top 20 bigrams
head(freq.df, 20)

# generating the wordcloud of the most popular bi-grams in triangular format 
wordcloud2(data=freq.df[1:200, ], size = 0.5, shape = 'triangle-forward', color=brewer.pal(10, "BrBG"))

# plotting a barchart of the 15 most popular bi-grams
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent bigrams")


# TRIGRAMS GENERATION
TrigramsTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram = TermDocumentMatrix(corp.ngrams,
                                control=list(stopwords = myStopwords,
                                             tolower = T,
                                             stopwords = T,
                                             removePunctuation = T, 
                                             removeNumbers = T,
                                             stripWhitespace = T,
                                             tokenize = TrigramsTokenizer))

freq = sort(rowSums(as.matrix(tdm.trigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)

# returning top 20 trigrams
head(freq.df, 20)



# generating the wordcloud of the most popular tri-grams  
wordcloud2(data=freq.df[1:200, ], size = 0.5, color=brewer.pal(10, "BrBG"))


# plotting a barchart of the 15 most popular tri-grams
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Trigrams") + ylab("Frequency") +
  ggtitle("Most frequent Trigrams")


# PHASE 4: SENTIMENT ANALYSIS


#POSITIVITY-NEGATIVITY ANALYSIS
library(SentimentAnalysis)

# checking the pos-neg using QDAP dictionary
checkSentiment <- function(x) {
  return (convertToDirection(analyzeSentiment(x))$SentimentQDAP)
}
sentiments <- checkSentiment(reviews$text)
df.sentiment <- as.data.frame(table(sentiments))

# plotting the barchart of positive, neutral, negative sentiments
ggplot(df.sentiment, aes(x = reorder(sentiments, -Freq), y=Freq, fill=sentiments)) + 
  geom_bar(stat="identity", color="black") + theme_minimal() + xlab("Sentiment types")

# checking the number of positive and negative sentiments
table(df.sentiment)


# DEEPER SENTIMENT ANALYSIS
library(syuzhet)

d<-get_nrc_sentiment(reviews$text)

# returning the identified associations before transformation
head(d,10)

#outcome transformation and grouping
td<-data.frame(t(d))
td1 <- data.frame(rowSums(td))
names(td1)[1] <- "count"
td1 <- cbind("sentiment" = rownames(td1), td1)
rownames(td_new) <- NULL
td2<-td1[1:8,]

# generating the barplot of sentiment (reactions) types
ggplot(td2, aes(x = reorder(sentiment, -count), y=count, fill=sentiment)) + 
  geom_bar(stat="identity", color="black") + theme_minimal()+xlab("Sentiment types")


