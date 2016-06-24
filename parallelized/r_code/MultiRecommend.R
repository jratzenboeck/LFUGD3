post.process <- function(predict.path, predictset) {
  # Post processing.
  # Insuring the original order
  predictset.original.order <- read.csv(predict.path,
                                        header = FALSE, 
                                        sep = "\t", 
                                        col.names = c("user", "item", "rating"), 
                                        na.strings = "?")
  predictset.original.order$id <- 1:nrow(predictset.original.order)
  
  predictset <- merge(predictset.original.order[c("user", "item", "id")], predictset, by = c("user", "item"), all = FALSE)
  predictset <- predictset[order(predictset$id), ]
  predictset$id <- NULL
  
  # For any rating greater than 5, set it to 5
  predictset$rating[predictset$rating > 5] <- 5
  
  # For any rating smaller than 1, set it to 1
  predictset$rating[predictset$rating < 1] <- 1
  
  return (predictset)
}

svd <- function(train.path, predict.path, predict.out.path, temp_file.path) {
  library(recosystem)
  
  # Building the recommender
  recommender <- Reco()
  recommender.training <- recommender$train(train.path, opts = c(dim = 50, lrate = 0.1, cost = 0.01, nthread = 20, niter = 100))
  recommender.prediction <- recommender$predict(predict.path, temp_file.path)
  
  # Reading the temorary prediction output into the original output
  predictions <- read.csv(temp_file.path,
                          header = FALSE,
                          col.names = c("rating"))
  predictset <- read.csv(predict.path,
                         header = FALSE,
                         sep = "\t",
                         col.names = c("user", "item", "rating"),
                         na.strings = "?")
  predictset$rating <- predictions$rating
  
  # Post processing.
  predictset <- post.process(predict.path = predict.path,
                             predictset = predictset)
  
  # Writing final output to CSV file
  write.table(predictset,
              file = predict.out.path,
              sep = "\t",
              quote = FALSE,
              row.names = FALSE,
              col.names = FALSE)
}

collborative_filtering <- function(train.path, predict.path, predict.out.path, item_based = TRUE) {
  library("recommenderlab")
  library("reshape2")
  
  dataset <- read.csv(train.path, 
                      header = FALSE, 
                      sep = "\t", 
                      col.names = c("user", "item", "rating"), 
                      na.strings = "?")
  
  predictset <- read.csv(predict.path,
                         header = FALSE, 
                         sep = "\t", 
                         col.names = c("user", "item", "rating"), 
                         na.strings = "?")
  
  # Adding text to IDs to make sure they don't mix up with indices
  dataset$user <- paste("u", dataset$user, sep = "")
  dataset$item <- paste("i", dataset$item, sep = "")
  predictset$user <- paste("u", predictset$user, sep = "")
  predictset$item <- paste("i", predictset$item, sep = "")
  
  # Calculate the mean rating per user
  dataset.user.mean.rating <- aggregate(dataset[, 3], list(dataset$user), mean)
  names(dataset.user.mean.rating)[1] <- "user"
  names(dataset.user.mean.rating)[2] <- "rating"
  
  # Convert to "realRatingMatrix"
  dataset.matrix <- as(dataset, "realRatingMatrix")
  
  # Creating the recommender
  if (item_based) {
    recommender <- Recommender(dataset.matrix, method = "IBCF", param = list(method = "Cosine", k = 1))
  } else {
    recommender <- Recommender(dataset.matrix, method = "UBCF", param = list(method = "Pearson", nn = 80))
  }
  
  # Predicting ratings for unrated items
  predictions.matrix <- predict(recommender, dataset.matrix, type = "ratings")
  predictions <- as(predictions.matrix, "matrix")
  
  # Creating a data frame from the prediction results
  predictions.data.frame <- melt(predictions, varnames = c("user", "item"), value.name = "rating")
  predictions.data.frame$user <- paste("u", predictions.data.frame$user, sep = "")
  
  # Setting the predicted ratings in the "predictset" data frame
  predictset <- merge(x = predictset[c("user", "item")], y = predictions.data.frame, all.x = TRUE)
  
  # Setting the not predicted ratings 
  # (which are the rating for the items that are not presented in the training set)
  # to the mean of each user's ratings
  predictset.rated <- predictset[complete.cases(predictset), ]
  predictset.not.rated <- predictset[!complete.cases(predictset), ]
  
  # Set the mean rating for not rated items
  predictset.not.rated <- merge(x = predictset.not.rated[c("user", "item")], y = dataset.user.mean.rating, by = "user", all.x = TRUE)
  
  # Get the final predictset
  predictset <- rbind(predictset.rated, predictset.not.rated)
  
  # Remove the extra text from IDs
  predictset$user <- paste(sub("u", "", predictset$user))
  predictset$item <- paste(sub("i", "", predictset$item))
  
  # Post processing.
  predictset <- post.process(predict.path = predict.path,
                             predictset = predictset)
  
  # Writing final output to CSV file
  write.table(predictset,
              file = predict.out.path,
              sep = "\t",
              quote = FALSE,
              row.names = FALSE,
              col.names = FALSE)
}

# train.path <- "Data/task03/training.dat"
# predict.path <- "Data/task03/predict.dat"
# 
# predict.out.path <- "Data/task03/predict.out.svd.dat"
# temp_file.path <- "Data/task03/predict.temp.dat"
# svd(train.path = train.path,
#     predict.path = predict.path,
#     predict.out.path = predict.out.path,
#     temp_file.path = temp_file.path)
# 
# predict.out.path <- "Data/task03/predict.out.item.dat"
# collborative_filtering(train.path = train.path,
#                        predict.path = predict.path,
#                        predict.out.path = predict.out.path,
#                        item_based = TRUE)
# 
# predict.out.path <- "Data/task03/predict.out.user.dat"
# collborative_filtering(train.path = train.path,
#                        predict.path = predict.path,
#                        predict.out.path = predict.out.path,
#                        item_based = FALSE)

files_number = 3
for (i in 1:files_number) {
  train.path <- paste("Data/task03/", i, ".rec.train", sep = "")
  predict.path <- paste("Data/task03/", i, ".rec.test", sep = "")
  
  predict.out.path <- paste("Data/task03/", i, ".cf.svd.out", sep = "")
  temp_file.path <- paste("Data/task03/", i, ".cf.svd.out.TEMP", sep = "")
  svd(train.path = train.path,
      predict.path = predict.path,
      predict.out.path = predict.out.path,
      temp_file.path = temp_file.path)
  
  predict.out.path <- paste("Data/task03/", i, ".cf.item.out", sep = "")
  collborative_filtering(train.path = train.path,
                         predict.path = predict.path,
                         predict.out.path = predict.out.path,
                         item_based = TRUE)
  
  predict.out.path <- paste("Data/task03/", i, ".cf.user.out", sep = "")
  collborative_filtering(train.path = train.path,
                         predict.path = predict.path,
                         predict.out.path = predict.out.path,
                         item_based = FALSE)
}

