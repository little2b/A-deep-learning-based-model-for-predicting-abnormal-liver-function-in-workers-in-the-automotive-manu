
normaldata <- read_excel("data_model.xlsx")
for (i in c(1:9)) {
  colnames(normaldata)[i] <- paste0("x",i)
}
colnames(normaldata)[10] <- "y"
#因子类型不能用数字
normaldata <- normaldata %>%
  mutate(x1=as.factor(x1),
         x4=as.factor(x4),
         x5=as.factor(x5),
         x9=as.factor(x9),
         y=as.factor(y))
normaldata <- normaldata %>%
  mutate(x1 = factor(x1, labels = make.names(levels(x1))),
         x4 = factor(x4, labels = make.names(levels(x4))),
         x5 = factor(x5, labels = make.names(levels(x5))),
         x9 = factor(x9, labels = make.names(levels(x9))),
         y = factor(y, labels = make.names(levels(y))))
set.seed(1234)
# normaldata$y <- as.factor(normaldata$y)
train_index <- createDataPartition(normaldata[[10]], p=0.7, list=F)
# normaldata$y <- as.numeric(normaldata$y)
trainset <- normaldata[train_index,]
testset <- normaldata[-train_index,]
#设置训练模式
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

model_xgboost = train(y ~ ., 
                 data=trainset, 
                 method='xgbDART', 
                 tuneLength=3,
                 trControl = fitControl)
#训练集混淆矩阵
xgboostpredictiontrain <- predict(model_xgboost,trainset)
xgboostcontrain <- confusionMatrix(table(xgboostpredictiontrain,trainset$y))
#测试集混淆矩阵
xgboostpredictiontest <- predict(model_xgboost,testset)
xgboostcontest <- confusionMatrix(table(xgboostpredictiontest,testset$y))
#训练集auc
xgboostpredictiontrain <- predict(model_xgboost,trainset,type="prob")
xgboostroctrain <- roc(trainset$y,xgboostpredictiontrain$X1,auc=TRUE,ci=TRUE)
#测试集AUC
xgboostpredictiontest <- predict(model_xgboost,testset,type="prob")
xgboostroctest <- roc(testset$y,xgboostpredictiontest$X1,auc=TRUE,ci=TRUE)

save(model_xgboost,file = "model_xgboost.RData")


