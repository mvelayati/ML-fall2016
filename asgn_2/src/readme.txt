The program was developed using Python as the programming language andd PyCharm as the IDE. 

# ===============
# Bernoulli Model
# ===============

To run the Bernoulli classification, comment out all the lines for the Multinomial approach. 


# =================
# Multinomial Model
# =================

To run the Multinomial classification, comment out all the lines for the Bernoulli approach.



To run the code for "Priors and overfitting" section, comment out the Bernoulli section entirely and also the Multinomial part except two last lines as shown below.
--multi = NaiveBayes(tweets_train, labels_train, tweets_dev, labels_dev, stopwords)
--multi.MAP_estimation() 