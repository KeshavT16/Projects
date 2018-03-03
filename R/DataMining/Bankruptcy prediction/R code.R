library("tidyverse")
library(ROCR)
library(rpart)

#data load
bank.data <- read.csv("bankruptcy.csv")

colnames(bank.data)

head(bank.data)

str(bank.data)

summary(bank.data)


###step 1
## EDA
## Link functions


#test-train data
subset <- sample(nrow(bank.data), nrow(bank.data) * 0.8)
bank.train = bank.data[subset, ]
bank.test = bank.data[-subset, ]


#modeling train data

#logistic regression
#probit
bank.glm0.probit<- glm(DLRSN ~ . - CUSIP -FYEAR,family = binomial(link = 'probit'), bank.train)
summary(bank.glm0.probit)
#AIC: 2421.1
#Residual deviance: 2399.1
#BIC


#logit
bank.glm0.logit <- glm(DLRSN ~ . - CUSIP -FYEAR, family = binomial, bank.train)
summary(bank.glm0.logit)
#AIC: 2415.5
#Residual deviance: 2393.5



#complementary log log
bank.glm0.clog <- glm(DLRSN ~ . - CUSIP -FYEAR, family = binomial(link="cloglog"), bank.train)
summary(bank.glm0.clog)
#AIC: 2434.5
#Residual deviance: 2412.5  on 4337  degrees of freedom

##Step 2  
## (a): Best model using AIC, BIC and lasso variable selection
## (b): Draw ROC
## (c): Report mean residual deviance, AUC, Misclassification rate 

bank.glm0.logit.nullmodel<- glm(DLRSN ~ 1, family = binomial, bank.train)
bank.glm0.logit.fullmodel <- glm(DLRSN ~ . - CUSIP -FYEAR, family = binomial, bank.train)

#AIC
bank.glm.AIC<- step(bank.glm0.logit.nullmodel, scope=list(lower=bank.glm0.logit.nullmodel, upper=bank.glm0.logit.fullmodel), direction='both', k = 2)
##DLRSN ~ R10 + R7 + R8 + R2 + R6 + R4 + R3 + R1 + R9
summary(bank.glm.AIC)
BIC(bank.glm.AIC)
##AIC: 2394.6
##Residual deviance: 2394.6  on 4338  degrees of freedom
##BIC: 2478.396
pred.bank.glm.AIC.test<- predict(bank.glm.AIC,new = bank.test, type="response")
pred.bank.glm.AIC.train<- predict(bank.glm.AIC, type="response")
pred <- prediction(pred.bank.glm.AIC.test, bank.test$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
#AUC: 0.8694456
#Misclassification Rate
# get binary prediction
pred.bank.train.b<- (pred.bank.glm.AIC.train > 1/16)*1
# get confusion matrix
table(bank.train$DLRSN, (pred.bank.glm.AIC.train > 1/16)*1, dnn = c("True", "Predicted"))
mean(bank.train$DLRSN != pred.bank.train.b)
# Misclassification rate :  0.3399264
# FNR :0.08306709
FNR <- sum(bank.train$DLRSN == 1 & pred.bank.train.b == 0)/sum(bank.train$DLRSN ==1)


#BIC
bank.glm.BIC <- step(bank.glm0.logit.nullmodel, scope=list(lower=bank.glm0.logit.nullmodel, upper=bank.glm0.logit.fullmodel), direction='both', k = log(nrow(bank.train))) #BIC
##Step:  AIC=2476.75
## residual deviance =2401/4339
##DLRSN ~ R10 + R7 + R8 + R2 + R6 + R4 + R3 + R1
summary(bank.glm.BIC)
BIC(bank.glm.AIC)
# BIC 2478.396
#insample
pred.bank.glm.BIC.train<- predict(bank.glm.BIC, type="response")
pred <- prediction(pred.bank.glm.BIC.train, bank.train$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
table(bank.train$DLRSN, (pred.bank.glm.BIC.train>1/16)*1, dnn = c("True", "Predicted"))
mean(bank.train$DLRSN != (pred.bank.glm.BIC.train>1/16)*1)


#out of sample
pred.bank.glm.BIC.test<- predict(bank.glm.BIC,new = bank.test, type="response")
pred <- prediction(pred.bank.glm.BIC.test, bank.test$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
#AUC: 0.8674271
#Misclassification Rate
# get binary prediction
pred.bank.test.b<- (pred.bank.glm.BIC.test > 0.06)*1
# get confusion matrix
table(bank.test$DLRSN, pred.bank.test.b, dnn = c("True", "Predicted"))
mean(bank.train$DLRSN != pred.bank.train.b)
# Misclassification rate :   0.3383165
# FNR :0.0798722
FNR <- sum(bank.train$DLRSN == 1 & pred.bank.train.b == 0)/sum(bank.train$DLRSN ==1)



#LASSO
bank.std<- scale(dplyr::select(bank.data,-DLRSN,-CUSIP,-FYEAR))
X.train<- as.matrix(bank.std)[subset,]
X.test<- as.matrix(bank.std)[-subset,]
Y.train<- as.numeric(as.matrix(bank.data)[subset,"DLRSN"])
Y.test <- as.numeric(as.matrix(bank.data)[-subset,"DLRSN"])
lasso.fit<- glmnet(x=X.train, y=Y.train, family = "gaussian", alpha = 1)

plot(lasso.fit, xvar = "lambda")


bank.glm.LASSSO <- step(bank.glm0, k = 2) #AIC
cv.lasso<- cv.glmnet(x=X.train, y=Y.train, family = "gaussian", alpha = 1, nfolds = 10)
plot(cv.lasso)
coef(lasso.fit, s=cv.lasso$lambda.1se)
## all vars except R1, R4, and R9
lasso.train.model <- glm(DLRSN ~ . - CUSIP -FYEAR -R1 -R4 -R9, family = binomial, bank.train)
summary(lasso.train.model)
pred.lasso.train<- predict(lasso.fit, newx=X.train, s=lasso.fit$lambda.1se, type = "response")
##AIC: 2441.2
##Residual deviance: 2425.2  on 4340
##BIC: 2492.24


##Step 4

costfunc.asym = function(obs, pred.p, pcut){
  weight1 = 15   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function
bank.final.model<- glm(DLRSN ~ . - CUSIP -FYEAR -R5 -R9, family = binomial, bank.train)

bank.model.predict.train <-  predict(bank.final.model, type="response")

p.seq = seq(0.01, 1, 0.01) 
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc.asym(obs = bank.train$DLRSN, pred.p = bank.model.predict.train, pcut = p.seq[i])  
}
plot(p.seq, cost)
optimal.pcut.glm0 = p.seq[which(cost==min(cost))]
## optimal probability : 0.41 0.42 0.45
## optimal probability : 0.06

## Step 5
library(boot)
pcut <- optimal.pcut.glm0
pcut <- 0.41#
pcut<- 1/16
costfunc = function(obs, pred.p){
  weight1 = 15   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
}

costfunc.sym = function(obs, pred.p){
  weight1 = 1   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
}

bank.final.model<- glm(DLRSN ~ R1+R2+R3+R4+R6+R7+R8+R10, family = binomial, bank.data)
summary(bank.final.model)
##mean square- 0.5566611

# on pcut as 1/16 - 0.3474982 0.3474611
# on p cut as optimal one - 0.1186159
# 0.3530169 0.3527953
cv.result<-cv.glm(data= bank.data,glmfit = bank.final.model, K = 12, cost = costfunc)
cv.result$delta
# misclassification rate with 0.06 optimal

# with pcut as 0.0625 0.4871229 0.4865364
#0.4941133 0.4945932



pred.bank.glm.BIC<- predict(bank.final.model, type="response")
pred <- prediction(pred.bank.glm.BIC, bank.data$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
#AUC: 0.8674271
0.8793651
#Misclassification Rate
# get binary prediction
pred.bank.train.b<- (pred.bank.glm.BIC > pcut)*1
# get confusion matrix
table(bank.data$DLRSN, (pred.bank.glm.BIC > pcut)*1, dnn = c("True", "Predicted"))
mean(bank.data$DLRSN != (pred.bank.glm.BIC > 0.4)*1)


## Step 6:
## (a): Fit CART (use 5:1 loss)
## (b): Report out-of-sample AUC and misclassification rate

bank_rpart<-rpart(DLRSN ~ . - CUSIP -FYEAR,bank.train,method = "class", parms = list(loss = matrix(c(0, 15, 1, 0), nrow = 2)))
bank.test.pred.tree1 = predict(bank_rpart, bank.test, type = "class")
bank.train.pred.tree1 = predict(bank_rpart, bank.train, type = "class")
table(bank.test$DLRSN, bank.test.pred.tree1, dnn = c("Truth", "Predicted"))
table(bank.train$DLRSN,bank.train.pred.tree1, dnn = c("Truth", "Predicted"))

## in sample misclassification rate
mean(bank.test.pred.tree1!= bank.test$DLRSN)
mean(bank.train.pred.tree1!= bank.train$DLRSN)

bank.test.prob.rpart3 = predict(bank_rpart, bank.train, type = "prob")
pred = prediction(bank.test.prob.rpart3[,2], bank.test$DLRSN)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]
##0.8101457
head(bank.test.prob.rpart2)
pred.bank.tree.value <- (bank.test.prob.rpart2[, 2]>0.06)*1
mean(pred.bank.tree.value!= bank.test$DLRSN)

#0.3860294


pred <- prediction(bank.test.prob.rpart3[,2], bank.train$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
