logisticmodel <- glm(y~.,family=binomial(link = 'logit'),data=trainset)

#训练集AUC
logpretrain <- predict.glm(logisticmodel,type='response',newdata=trainset)
logtrainroc <- roc(trainset$y,logpretrain,auc=TRUE,ci=TRUE)
######创建函数，R语言求解cutoff, AUC, 95%置信区间，敏感性，特异性
cal_metrics <- function(label, pred){
  roc.p=pROC::roc(label, pred,ci =T)
  if (roc.p$auc>0.5){
    cutoff=roc.p$thresholds[which.max(roc.p$sensitivities+roc.p$specificities)]
    sensitivity=roc.p$sensitivities[which.max(roc.p$sensitivities+roc.p$specificities)]
    specificity=roc.p$specificities[which.max(roc.p$sensitivities+roc.p$specificities)]
    ci = roc.p$ci
    df=data.frame(type='positive classification',
                  auc=round(roc.p$auc,3),cutoff=cutoff,
                  sensitivity=sensitivity,specificity=specificity,ci_lower = ci[1],ci_upper=ci[3])
    return(df)
  }
  else{
    cutoff=roc.p$thresholds[which.min(roc.p$sensitivities+roc.p$specificities)]
    sensitivity=roc.p$sensitivities[which.min(roc.p$sensitivities+roc.p$specificities)]
    specificity=roc.p$specificities[which.min(roc.p$sensitivities+roc.p$specificities)]
    ci = roc.p$ci
    df=data.frame(type='negative classification',
                  auc=1-round(roc.p$auc,3),cutoff=cutoff,
                  sensitivity=1-sensitivity,specificity=1-specificity,ci_lower = ci[1],ci_upper=ci[3])
    return(df)
  }
}
cal_metrics(trainset$y,logpretrain)
#训练集混淆矩阵
logtrain <- as.factor(ifelse(logpretrain>0.152,"X1","X0")) %>% 
  factor(levels = c("X0","X1"))
trainset$y <- factor(trainset$y,levels = c("X0","X1"))
logcontrain <- confusionMatrix(table(logtrain,trainset$y))
#测试集AUC
logpretest <- predict.glm(logisticmodel,type='response',newdata=testset)
logtestroc <- roc(testset$y,logpretest,auc=TRUE,ci=TRUE)
#测试集混淆矩阵
logtest <- ifelse(logpretest>0.152,"X1","X0")
logcontest <- confusionMatrix(table(logtest,testset$y))

save(logisticmodel,file = "logisticmodel.RData")





