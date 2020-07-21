#Three popular female novelists all born in the 1850s: 17 Helen Mathers 1853-1920 (18010- 18669 in the kaggle csv file), 32 Lucas Malet 1852-1931 (33861-34563), 33 Marie Corelli 1855-1924 (34564-36305)
#200 lines each
#there is a â in the code. If this code is loaded to RStudio, the encoding of it should be changed to UTF-8!!!
#obtained from DS7003 coursework and then added the deep learning part to the end

#set working directory and load package tm
setwd(dirname(file.choose()))
getwd()
if (!require('tm')) install.packages('tm'); library('tm')

#input data and form three dataframes
if(!file.exists('Gungor_2018_VictorianAuthorAttribution_data-train.csv')){
	download.file('http://archive.ics.uci.edu/ml/machine-learning-databases/00454/dataset.zip', 'dataset.zip')
	unzip('dataset.zip')
	file.copy('./dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv', '.')
#if the working directory does not have the csv file, this if statement
#needs several minutes to run
}
dfVictorianEraAA <- read.table('Gungor_2018_VictorianAuthorAttribution_data-train.csv', header = TRUE, sep = (','))
dfHelen_Mathers18009_18208 <- dfVictorianEraAA[18009:18208,]
dfLucas_Malet33860_34059 <- dfVictorianEraAA[33860:34059,]
dfMarie_Corelli34563_34762 <- dfVictorianEraAA[34563:34762,]

#form corpa from dataframes.
#texts are already all in lower case and no punctuation
#package tm is required
dfHelen_Mathers18009_18208_corpus <- VCorpus(VectorSource(dfHelen_Mathers18009_18208$text))
dfHelen_Mathers18009_18208_corpus <- tm_map(dfHelen_Mathers18009_18208_corpus, stripWhitespace)
dfLucas_Malet33860_34059_corpus <- VCorpus(VectorSource(dfLucas_Malet33860_34059$text))
dfLucas_Malet33860_34059_corpus <- tm_map(dfLucas_Malet33860_34059_corpus, stripWhitespace)
dfMarie_Corelli34563_34762_corpus <- VCorpus(VectorSource(dfMarie_Corelli34563_34762$text))
dfMarie_Corelli34563_34762_corpus <- tm_map(dfMarie_Corelli34563_34762_corpus, stripWhitespace)

#form dtm. Each line a document (1000 words)
#change minimum word length to 1 from 3
dfHelen_Mathers18009_18208_dtDf <- as.data.frame(as.matrix(DocumentTermMatrix(dfHelen_Mathers18009_18208_corpus, control=list(wordLengths = c(1, Inf)))))
dfLucas_Malet33860_34059_dtDf <- as.data.frame(as.matrix(DocumentTermMatrix(dfLucas_Malet33860_34059_corpus, control=list(wordLengths = c(1, Inf)))))
dfMarie_Corelli34563_34762_dtDf <- as.data.frame(as.matrix(DocumentTermMatrix(dfMarie_Corelli34563_34762_corpus, control=list(wordLengths = c(1, Inf)))))

#retain only columns of words which can found both in HM, LM and MC's texts
common_cols <- intersect(intersect(colnames(dfHelen_Mathers18009_18208_dtDf), colnames(dfLucas_Malet33860_34059_dtDf)), colnames(dfMarie_Corelli34563_34762_dtDf))
HmLmMcDtDf <- rbind(dfHelen_Mathers18009_18208_dtDf[common_cols], dfLucas_Malet33860_34059_dtDf[common_cols], dfMarie_Corelli34563_34762_dtDf[common_cols])#5228 cols

#further retain only columns of words each of which are at least appeared
#300 times 0.05%
HmLmMcTtl300OrMore <- HmLmMcDtDf[, colSums(HmLmMcDtDf) >=300] #237
#texts quite untidy. number of â in HM 2077, LM 1743 and MC 6280
#delete the column â
HmLmMcTtl300OrMore$â <- NULL #236

#aggreate and sum every four lines (reduced to 150 lines)
#add and delete column textNO
HmLmMcTtl300OrMore$textNo <- rep(1:150, each = 4)
dfHmLmMcWdFeqDf <- aggregate(. ~ textNo, HmLmMcTtl300OrMore, sum)
dfHmLmMcWdFeqDf$textNo <- NULL

#add labels HM, LM and MC and put the column to the front
dfHmLmMcWdFeqDf$HmOrLmOrMc <- c(rep('HM', 50), rep('LM', 50), rep('MC', 50))
dfHmLmMcWdFeqDfLabled = dfHmLmMcWdFeqDf[,c(237,1:236)] #236+1

#shuffling rows:
set.seed(12345)
rrowNos <- sample(nrow(dfHmLmMcWdFeqDfLabled))
dfHmLmMcWdFeqDfLabledRandm <- dfHmLmMcWdFeqDfLabled[rrowNos,]

#normalisation
data_norm <- function(x) {(x- min(x))/ (max(x)- min(x))}
dfHmLmMcWdFeqDfLabledRandm_norm <- as.data.frame(lapply(dfHmLmMcWdFeqDfLabledRandm[,-1], data_norm))
summary(dfHmLmMcWdFeqDfLabledRandm_norm[,1:4]) #see whether normalised

#KNN!
if (!require('class')) install.packages('class'); library('class')
dfHmLmMcWdFeqDfLabledRandm_norm_train <- dfHmLmMcWdFeqDfLabledRandm_norm[1:120,]
dfHmLmMcWdFeqDfLabledRandm_norm_test <- dfHmLmMcWdFeqDfLabledRandm_norm[121:150,]
HmOrLmOrMc_pred <- knn(dfHmLmMcWdFeqDfLabledRandm_norm_train, dfHmLmMcWdFeqDfLabledRandm_norm_test, dfHmLmMcWdFeqDfLabledRandm[1:120,1], k= 11)
table(pred = HmOrLmOrMc_pred, true_HelenMathers_LucasMalet_MarieCorelli_KNN = dfHmLmMcWdFeqDfLabledRandm[121:150,1]) #mistake rate 1/30
#sqrt(120) = 10.954 . Therefore use k =11. 
#k = 11 perform the best, only one error: 1 MC was misjudged as LM

#SVM! tune automatically
if (!require('e1071')) install.packages('e1071'); library('e1071')library
HmOrLmOrMc_svm_model <- svm(dfHmLmMcWdFeqDfLabledRandm_norm_train, dfHmLmMcWdFeqDfLabledRandm[1:120,1], type = 'C')
pred <- predict(HmOrLmOrMc_svm_model, dfHmLmMcWdFeqDfLabledRandm_norm_test)
table(pred, true_HelenMathers_LucasMalet_MarieCorelli_SVM = dfHmLmMcWdFeqDfLabledRandm[121:150,1])
#all correct

#tune manually
dfHmLmMcWdFeqDfLabledRandm1To120AsFactors = as.factor(dfHmLmMcWdFeqDfLabledRandm[1:120,1])
set.seed(12345)
svm_tune <- tune(svm, train.x = dfHmLmMcWdFeqDfLabledRandm_norm_train,
						train.y = dfHmLmMcWdFeqDfLabledRandm1To120AsFactors,
						kernel = 'linear',
						type = 'C',
						ranges = list(cost = c(.001,.01,.1,1,5,10,100)))
svm_tune 
svm_tune$best.model

#besides best cost, also best number of support vectors, etc.
pred_svm_after_tune <- predict(svm_tune$best.model, dfHmLmMcWdFeqDfLabledRandm_norm_test)
table(pred = pred_svm_after_tune, true_HelenMathers_LucasMalet_MarieCorelli_TunedSVM = dfHmLmMcWdFeqDfLabledRandm[121:150,1])

# Deep learning using package keras
# import keras
# note: use_condaenv("r_reticulate") is only work for my Asus PC + Windows 10
# see README.md for matters related to installation of Python
if (!require('keras')) install.packages('keras'); library('keras')
use_condaenv("r_reticulate")

# Convert to matrix
training <- as.matrix(dfHmLmMcWdFeqDfLabledRandm_norm[1:120,])
dimnames(training) <- NULL
test <- as.matrix(dfHmLmMcWdFeqDfLabledRandm_norm[121:150,])
dimnames(test) <- NULL

# Convert labels to numerics and one hot encoding form
trainLabels <- to_categorical(as.numeric(as.factor(dfHmLmMcWdFeqDfLabledRandm[1:120,1])) - 1)
testtarget <- as.numeric(as.factor(dfHmLmMcWdFeqDfLabledRandm[121:150,1])) - 1
testLabels <- to_categorical(testtarget)

# Create sequential model (236 input columns, 3 categories)
model <- keras_model_sequential()
model %>% #one hidden layer, units = 79 (236 input columns, 3 categories)
         layer_dense(units=79, activation = 'relu', input_shape = c(236)) %>%
         layer_dense(units = 3, activation = 'softmax')
summary(model)

# Compile
model %>%
         compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = 'accuracy')

# Fit model				 
history <- model %>%
	 fit(training,
		 trainLabels,
		 epoch = 200,
		 batch_size = 32,
		 validation_split = 0.2)

# Prediction & confusion matrix - test data and labels
pred <- model %>%
		predict_classes(test)

library(caret)
confusionMatrix(table <- table(Predicted = pred, Actual = testtarget), mode = "everything")       

		 
