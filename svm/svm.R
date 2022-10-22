
fitControl <- trainControl(
  method = 'cv',                  # 重复交叉验证
  number = 10,                    # 10折交叉
  search = "grid",                #网格寻优
  savePredictions = 'final',      # saves predictions for optimal tuning parameter 
  classProbs = T,                 # should class probabilities be returned
  summaryFunction=twoClassSummary,
  sampling = "down",
  allowParallel = TRUE
) 

model_svmRadial = train(y ~ ., 
                        data=trainset, 
                        method='svmRadial', 
                        tuneLength=15, 
                        metric="Accuracy",
                        trControl = fitControl)
#训练集混淆矩阵
svmpredictiontrain <- predict(model_svmRadial,trainset)
svmcontrain <- confusionMatrix(table(svmpredictiontrain,trainset$y))
#测试集混淆矩阵
svmpredictiontest <- predict(model_svmRadial,testset)
svmcontest <- confusionMatrix(table(svmpredictiontest,testset$y))
#训练集auc
svmpredictiontrain <- predict(model_svmRadial,trainset,type="prob")
svmroctrain <- roc(trainset$y,svmpredictiontrain$X1,auc=TRUE,ci=TRUE)
#测试集AUC
svmpredictiontest <- predict(model_svmRadial,testset,type="prob")
svmroctest <- roc(testset$y,svmpredictiontest$X1,auc=TRUE,ci=TRUE)

save(model_svmRadial,file = "model_svmRadial.RData")
