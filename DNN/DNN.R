library(tidyverse)
library(readxl)
library(neuralnet)
library(foreach)
library(doParallel)
library(deepnet)
library(caret)
library(pROC)
cores <- detectCores()
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
normaldata <- read_excel("data_model.xlsx")

for (i in c(1:9)) {
  colnames(normaldata)[i] <- paste0("x",i)
}
colnames(normaldata)[10] <- "y"

set.seed(1234)
normaldata$y <- as.factor(normaldata$y)
train_index <- createDataPartition(normaldata[[10]], p=0.7, list=F)
normaldata$y <- as.numeric(normaldata$y)
trainset <- normaldata[train_index,]
testset <- normaldata[-train_index,]
#设置训练模式
fitControl <- trainControl(
  method = 'cv',                  # 重复交叉验证
  number = 10,                    # 10折交叉
  search = "grid",                #网格寻优
  savePredictions = 'final',      # saves predictions for optimal tuning parameter 
  #classProbs = T,                 # should class probabilities be returned
  summaryFunction=defaultSummary,
  #sampling = "down",
  allowParallel = TRUE
) 
#设置寻优参数
sadnnGrid <-  expand.grid(layer1 = c(1:5),
                        layer2 = c(1:5),
                        layer3 = c(1:5))
set.seed(6655)
model_nn = train(y ~ ., 
                 data=trainset, 
                 method='neuralnet', 
                 tuneGrid=sadnnGrid,
                 trControl = fitControl,
                 metric="AUC")

save(model_nn,file = "sadnn.RData")
stopImplicitCluster()


