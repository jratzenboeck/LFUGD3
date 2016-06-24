import clr
import csv
#clr.AddReference("MyMediaLite.dll")
#clr.AddReferenceToFileAndPath("E:\\Libraries\\C#\\MyMediaLite-3.11\\lib\\mymedialite\\MyMediaLite.dll")
#Please adapt
clr.AddReferenceToFileAndPath("J:\\SoftwareEntwicklung_Desktop\\IronPython\\LFUGD\\gSVG++\\gSVG++\\lib\\MyMediaLite-3.11\\lib\\mymedialite\\MyMediaLite.dll")
from MyMediaLite import *

# load the data
train_data = IO.RatingData.Read("data\u1.base")
test_data  = IO.RatingData.Read("data\u1.test")
atr_data  = IO.RatingData.Read("data\u.genre")

# set up the recommender
#recommender = RatingPrediction.UserItemBaseline() # don't forget ()
recommender = RatingPrediction.GSVDPlusPlus() # don't forget ()
#recommender = RatingPrediction.SVDPlusPlus() # don't forget ()
#recommender.Ratings = train_data
recommender.MinRating = 1
recommender.MaxRating = 5
recommender.NumFactors = 50
recommender.Regularization = 1
recommender.BiasReg = 0.005
recommender.BiasLearnRate = 0.07
recommender.LearnRate = 0.01
recommender.NumIter = 50
recommender.FrequencyRegularization = 1
recommender.AdditionalFeedback = atr_data
#recommender.ItemAttributes = atr_data
recommender.Ratings = train_data
recommender.Train()

# measure the accuracy on the test data set
#print Eval.Ratings.Evaluate(recommender, test_data)
#print Eval.RatingsCrossValidation(recommender, 10, 0, 1)

results = Eval.Ratings.Evaluate(recommender, test_data)

#recommender.WritePredictions(ratings, user_mapping, item_mapping, "test.csv");

# make a prediction for a certain user and item
print recommender.Predict(2, 2)

print results

#predWriter = csv.writer(open('predict.csv', 'wb'), delimiter=' ')

#predWriter.writerow()