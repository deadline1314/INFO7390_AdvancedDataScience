#######################################################################################
######### 		R-script for Loan Granting Binary Classification
######### 		Path to RawData: /Users/rahelehhassanisadi/Amazon Drive/Documents/Ebrahim/Fall_2017_7390/Lec_5_LogisticRegression/workspace/data/rawdata/ 
######### 		Path to Output Files: /Users/rahelehhassanisadi/Amazon Drive/Documents/Ebrahim/Fall_2017_7390/Lec_5_LogisticRegression/workspace/data/output
#########     Path to Functions : /Users/rahelehhassanisadi/Amazon Drive/Documents/Ebrahim/Fall_2017_7390/Lec_5_LogisticRegression/workspace/src
#########     How to run this script from command line: /usr/local/bin/Rscript main_V1.0.r 
#######################################################################################

#################################
## Required packages
#################################
require(data.table)
library(VIM)
library(mice)
library(caret)
library(dplyr)
library(corrplot)
library(ROCR)
library(pROC)
#library(randomForest)

#################################
## Required functions
#################################
#set the directory to the folder containing functions
setwd('/Users/rahelehhassanisadi/Amazon Drive/Documents/Ebrahim/Fall_2017_7390/Lec_5_LogisticRegression/workspace/src')
source('functions_V1.0.R')

#################################
## Input and output directories
#################################
in_dir <- '/Users/rahelehhassanisadi/Amazon Drive/Documents/Ebrahim/Fall_2017_7390/Lec_5_LogisticRegression/workspace/data/rawdata'
out_dir <- '/Users/rahelehhassanisadi/Amazon Drive/Documents/Ebrahim/Fall_2017_7390/Lec_5_LogisticRegression/workspace/data/output'

#################################
## Read rawdata
#################################
setwd(in_dir)
df <- fread('Loan Granting Binary Classification.csv', header = T, stringsAsFactors = TRUE)

#################################
## Data cleaning
#################################

# Convert Monthly_Debt to numeric values
df$Monthly_Debt <- as.numeric(gsub(",", "", df$Monthly_Debt))

# Remove duplicated raws
setkey(df, Loan_ID, Customer_ID)
df <- unique(df)

# Missing Data Patterns
setwd(out_dir)
write.table(md.pattern(df), 'Missing_Data_Pattern.txt',
            sep="\t", row.names=FALSE)

graphics.off()	## Close all plot windows
pdf('Missing_Data_Pattern.pdf', onefile=TRUE)
aggr_plot <- aggr(df, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(df), cex.axis=.5, gap=3, ylab=c("Histogram of missing data","Pattern"))
dev.off()
graphics.off()	## Close all plot windows

# remove unnecessary columns
df <- within(df, rm(Loan_ID, Customer_ID, Current_Loan_Amount))

# Generates multiple imputations for incomplete multivariate data by Chained Equations (MICE) using Predictive mean matching
imp <- mice(data = df, m = 2, maxit = 1, method = "pmm")

# fills in the missing data and returns the completed data
#c_df <- complete(imp, "long", include = FALSE)
completed_df<- complete(imp, action = 1, include = FALSE)

write.table(completed_df, "Completed_data.txt",
            sep="\t", row.names=FALSE)
df <- completed_df

capture.output(summary(df), file = "Basic_Statistics.txt")
#################################
## plot Correlation Matric
#################################
plotCorrelationMatric(df)

#################################
## Train-test splitting
#################################
# 80% of samples -> fitting
# 20% of samples -> testing


#df <- droplevels(df)
df = as.data.frame(df)
df$Loan_Status = as.factor(df$Loan_Status)
index <- createDataPartition(df$Loan_Status, p=0.8, list=FALSE)
training <- df[ index, ]
testing <- df[ -index, ]
levels(training$Loan_Status) <- c("Charged_Off", "Fully_Paid")
levels(testing$Loan_Status) <- c("Charged_Off", "Fully_Paid")

#################################
## ROC Curve for predicting Class with logistic regression
#################################
setwd(out_dir)

train_control <- trainControl(method="cv", number=5,  classProbs = TRUE)
model_glm <- train(Loan_Status~., data= training, trControl=train_control, method="glm",  metric = "ROC")
model_name <- 'Logistic Regression'
capture.output(performanceAccuracy(model_glm, testing), file =  sprintf("AccuracyMetrics_%s.txt",model_name))
