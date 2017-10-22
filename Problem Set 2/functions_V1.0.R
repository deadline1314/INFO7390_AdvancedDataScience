#################################
## Define the required functions
#################################


#To compute the matrix of p-value, a custom R function is used :
# mat : is a matrix of data
# ... : further arguments to pass to the native R cor.test function
cor.mtest <- function(mat, ...) {
    mat <- as.matrix(mat)
    n <- ncol(mat)
    p.mat<- matrix(NA, n, n)
    diag(p.mat) <- 0
    for (i in 1:(n - 1)) {
        for (j in (i + 1):n) {
            tmp <- cor.test(mat[, i], mat[, j], ...)
            p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
        }
    }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

plotCorrelationMatric <- function(df){
    dataset <- select_if(df, is.numeric)
    M=cor(dataset)
    p.mat <- cor.mtest(dataset)
    
    graphics.off()	## Close all plot windows
    fig_title="Correlation Matrix"
    pdf(paste(fig_title,".pdf",sep=""), onefile=TRUE)
    corrplot(M, type="upper", 
                  addCoef.col = "white", # Add coefficient of correlation
                  tl.col="black", tl.srt=45, #Text label color and rotation
                                        #p.mat = p.mat, sig.level = 0.01
                , title='Correlation Matrix'
                                        # hide correlation coefficient on the principal diagonal
                                        #,diag=FALSE 
                 ,mar=c(0,0,1,0)
    )
    dev.off()
    graphics.off()	## Close all plot windows
}


performanceAccuracy <- function(model, df_testset){
    print(model)
    model_name <- 'Logistic Regression'
    result_predicted_prob <- predict(model, df_testset, type="prob") # Prob Prediction
    result_roc <- roc(df_testset$Loan_Status, result_predicted_prob$Fully_Paid) # Draw ROC curve.
    result_coords <- coords(result_roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy", "1-specificity", "sensitivity"))
    best_thr = round(result_coords['threshold'], digit = 2 )
    result_predicted_class <- as.factor(ifelse(result_predicted_prob$Fully_Paid >= best_thr, 'Fully_Paid', 'Charged_Off'))
    result_confusionMatrix <- confusionMatrix(data = df_testset$Loan_Status, reference = result_predicted_class, mode = "prec_recall", positive = "Fully_Paid")
    FPR = round(1 - result_confusionMatrix[["byClass"]]['Neg Pred Value'], digits=2)
    TPR = round(result_confusionMatrix[["byClass"]]['Pos Pred Value'], digits=2)
    print(result_confusionMatrix)
    print(result_coords)
    print(result_roc['auc'])

    graphics.off()	## Close all plot windows
    fig.title<-sprintf("%s_ROCCurve.pdf",model_name) 
    pdf(fig.title, onefile = TRUE) #, width=16, height=8.5)
    ROCRpred<-prediction(result_predicted_prob$'Fully_Paid', df_testset$Loan_Status)
    plot(performance(ROCRpred,'tpr','fpr'), col="black", main = paste("Model: ",model[['modelInfo']]['label'], "\nArea under the curve (AUC):", round(as.numeric(result_roc['auc']), digits=2), sep=" "), xlab = "Fales Positive Rate", ylab = "True Positive Rate")

    segments(0,0, 1,1, col="pink")
    
    i = 0.2
    while (i <= 0.9 ){
        thr = i + best_thr - round(best_thr, digit =1)
        result_predicted_class <- as.factor(ifelse(result_predicted_prob$'Fully_Paid' >= thr, 'Fully_Paid', 'Charged_Off'))
        result_confusionMatrix <- confusionMatrix(data = df_testset$Loan_Status, reference = result_predicted_class, mode = "prec_recall", positive = "Fully_Paid")
        FPR = round(1 - result_confusionMatrix[["byClass"]]['Neg Pred Value'], digits=2)
        TPR = round(result_confusionMatrix[["byClass"]]['Pos Pred Value'], digits=2)
        if (thr != best_thr){
            points(FPR, TPR, cex=1,pch=19,col="blue")
        } else {
            points(FPR, TPR, cex=1,pch=19,col="red")
        }
        text(FPR+0.08, TPR-0.02, labels=paste(thr , " (",FPR, ", ", TPR, ")", sep =""), cex= 0.7)
        i = i + 0.1
    }
    dev.off()
    graphics.off()	## Close all plot windows

    thr = 0.05
    df_threshold <- data.frame()
    while (thr < 1 ){
        result_predicted_class <- as.factor(ifelse(result_predicted_prob$Fully_Paid >= thr,'Fully_Paid', 'Charged_Off'))
        result_confusionMatrix <- confusionMatrix(data = df_testset$Loan_Status, reference = result_predicted_class, mode = "prec_recall", positive = "Fully_Paid")
        FPR = round(1 - result_confusionMatrix[["byClass"]]['Neg Pred Value'], digits=2)
        TPR = round(result_confusionMatrix[["byClass"]]['Pos Pred Value'], digits=2)
        df_threshold <- rbind(df_threshold, data.table(THRESHOLD=c(thr), FPR = c(FPR), TPR = c(TPR)))
        thr = thr + 0.01
    }
    print(df_threshold)
}


