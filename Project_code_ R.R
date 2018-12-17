library(leaps)
library(rpart)
library(rpart.plot)  			# Enhanced tree plots
library(rattle)           # Fancy tree plot
library(RColorBrewer) 
library(modeest)
library(stats)
library(caret)
library(utility)
library(DMwR)
library(randomForest)
library(unbalanced)
library("neuralnet")
library(class)

# Used this function on main dataset for variable reduction. 
?regsubsets()


########### #########################    Reading Dataset   ##########################################


rm(list = ls())
H1B<-read.csv("C:\\Users\\aysha\\Documents\\Daily Schedule\\2nd Semester\\Cs513\\Project\\1. Master H1B Dataset1_24apr.csv", na.strings =c("",NaN,NA))
H1B<- na.omit(H1B)

H1B_imp<-H1B[,-c(2,4)]

H1B_imp<- na.omit(H1B)
###########################################################  Encoding   #########################################################################

#Encoding to 1,2 

levels(H1B_imp$FULL_TIME_POSITION)<- list("1"="Y","2"="N")


# Encoding to 1,0

levels(H1B_imp$H.1B_DEPENDENT)<- list("1"="Y","0"="N")

# Encoding to 1,0

levels(H1B_imp$WILLFUL_VIOLATOR)<- list("1"="Y","0"="N")

# Dividing state region wise

levels(H1B_imp$EMPLOYER_STATE) <- list("1"=c("CT", "ME","MA","NH","RI","VT"),"2"=c("NJ", "NY", "PA"),"3"=c("IL","IN","MI","OH","WI"),"4"=c("IA","KS","MN","MO","NE","ND","SD"),"5"=c("DE","FL","GA","MD","NC","SC","VA","DC"),"6"=c("AL","KY","MS","TN")
                                   ,"7"=c("AR","LA","OK","TX"),"8"=c("AZ","CO","ID","MT","NV","NM","UT","WY"),"9"=c("AK","CA","HI","OR","WA"))

# Dividing Worksite regoin wise

levels(H1B_imp$WORKSITE_STATE) <- list("1"=c("CT", "ME","MA","NH","RI","VT"),"2"=c("NJ", "NY", "PA"),"3"=c("IL","IN","MI","OH","WI"),"4"=c("IA","KS","MN","MO","NE","ND","SD"),"5"=c("DE","FL","GA","MD","NC","SC","VA","DC"),"6"=c("AL","KY","MS","TN")
                                   ,"7"=c("AR","LA","OK","TX"),"8"=c("AZ","CO","ID","MT","NV","NM","UT","WY"),"9"=c("AK","CA","HI","OR","WA"))

# Encoding occupation into 8(1-8) sectors

levels(H1B_imp$SOC_NAME) <- list("1"=c("ANALYSTS", "COMPUTER OCCUPATION", "IT MANAGERS", "GRAPHIC DESIGNERS" ,"DESIGNERS")
                             ,"2"=c("MARKETING","COMMUNICATIONS","CURATORS","MANAGERS","PUBLIC RELATIONS","MANAGEMENT","HUMAN RESOURCES","FIRST LINE SUPERVISORS")
                             ,"3"=c("AGRICULTURE","ANIMAL HUSBANDARY"),"4"=c("BUSINESS OPERATIONS SPECIALIST"),"5"=c("FINANCE","ACCOUNTANTS","INSURANCE"),"6"=c("MATHEMATICIANS AND STATISTICIANS","ACTUARIES","ECONOMISTS","WRITERS EDITORS AND AUTHORS","INTERPRETERS AND TRANSLATORS","LIBRARIANS","EDUCATION"),"7"=c("REPORTERS AND CORRESPONDENTS","ARCHITECTURE","SOCIAL WORKERS","CONSTRUCTION","EVENT PLANNERS","SURVEYORS","TRANSPORTATION","MECHANICS","HISTORIANS","COACHES AND SCOUTS","LAB TECHNICIANS","LAWYERS AND LEGAL SUPPORT WORKERS","LOGISTICIANS","COUNSELORS","MULTIMEDIA ARTISTS AND ANIMATORS","RELIGIOUS WORKERS","ENTERTAINMENT","FASHION DESIGNERS","REAL ESTATE"),"8"=c("FITNESS TRAINERS","DOCTORS","HEALTHCARE","INTERNIST","FOOD PREPARATION WORKERS"))


# Encoding Case_status into 2(1,0) categories

levels(H1B_imp$CASE_STATUS)<- list("1"=c("CERTIFIED","CERTIFIEDWITHDRAWN"), "0"=c("DENIED","WITHDRAWN"))

# Standardizing instead of normalizing so that we can deal with outlier also
str(H1B)
H1B_imp$WAGE_YEARLY <- (H1B_imp$WAGE_YEARLY - mean(H1B_imp$WAGE_YEARLY)) / sd(H1B_imp$WAGE_YEARLY)  

#########################################  Encoding  ##################################################################


############################################## Dividing dataset into 2016  ###################################################3

H1B_2016<-H1B_imp[H1B_imp$CASE_SUBMITTED_YEAR==2016,]

################################################ Converting into Numeric  ##########################################

str(H1B_2016)


H1B_2016$EMPLOYER_STATE<-as.integer(H1B_2016$EMPLOYER_STATE)
H1B_2016$EMPLOYER_COUNTRY<-as.integer(H1B_2016$EMPLOYER_COUNTRY)
H1B_2016$SOC_NAME<-as.integer(H1B_2016$SOC_NAME)
H1B_2016$FULL_TIME_POSITION<-as.integer(H1B_2016$FULL_TIME_POSITION)
H1B_2016$H.1B_DEPENDENT<-as.integer(H1B_2016$H.1B_DEPENDENT)
H1B_2016$WORKSITE_STATE<-as.integer(H1B_2016$WORKSITE_STATE)
H1B_2016$CASE_STATUS<-as.integer(H1B_2016$CASE_STATUS)
H1B_2016$WILLFUL_VIOLATOR<-as.integer(H1B_2016$WILLFUL_VIOLATOR)



############################################# Training and Test ######################################################

set.seed(123)
index<-sort(sample(nrow(H1B_2016),round(.30*nrow(H1B_2016))))
training<-H1B_2016[-index,]
test<-H1B_2016[index,]
training<- na.omit(training)


# tried with model matrix as attributes type factor since encoding as numeric has impact on Neural net accuracy. But in the code shown only with numeric
#m_2017<-model.matrix(~CASE_STATUS+EMPLOYER_STATE+SOC_NAME+TOTAL_WORKERS+FULL_TIME_POSITION+H.1B_DEPENDENT+WILLFUL_VIOLATOR+WORKSITE_STATE+WAGE_YEARLY, data=training)


net_bc2_2016<- neuralnet(CASE_STATUS~EMPLOYER_STATE+SOC_NAME+TOTAL_WORKERS+FULL_TIME_POSITION+H.1B_DEPENDENT+WILLFUL_VIOLATOR+WORKSITE_STATE+WAGE_YEARLY,training,hidden=1,threshold=0.01)


#Plot the neural network
plot(net_bc2_2016)

str(test[,c(-1,-11)])

net_bc2_results <-compute(net_bc2_2016, test[,c(-1,-11)])
ANN=as.numeric(net_bc2_results$net.result)
ANN_round<-round(ANN)
ANN_cat<-ifelse(ANN<2.5,2,4)
table(Actual=test$Class,ANN_cat)
wrong<- (test$Class!=ANN_cat)
rate<-sum(wrong)/length(wrong)



# Using original dataset for KNN since column were removed with 

H1B$EMPLOYER_STATE<-as.numeric(H1B$EMPLOYER_STATE)
H1B$EMPLOYER_COUNTRY<-as.numeric(H1B$EMPLOYER_COUNTRY)
H1B$SOC_NAME<-as.numeric(H1B$SOC_NAME)
H1B$FULL_TIME_POSITION<-as.numeric(H1B$FULL_TIME_POSITION)
H1B$H.1B_DEPENDENT<-as.numeric(H1B$H.1B_DEPENDENT)
H1B$WORKSITE_STATE<-as.numeric(H1B$WORKSITE_STATE)
H1B$CASE_STATUS<-as.factor(H1B$CASE_STATUS)
H1B$WILLFUL_VIOLATOR<-as.numeric(H1B$WILLFUL_VIOLATOR)
H1B$VISA_CLASS<- as.numeric(H1B$VISA_CLASS)

####2016 Data ####################
H1B_2016<-H1B[H1B$CASE_SUBMITTED_YEAR==2016,]


set.seed(123)
index<-sort(sample(nrow(H1B_2016),round(.30*nrow(H1B_2016))))
training<-H1B_2016[-index,]
test<-H1B_2016[index,]
training<- na.omit(training)
test<- na.omit(test)
str(training)

training[,3]

# Knn

# Classifying based on CASE_STATUS
Predict_h1b_2016 <- knn(training[,-11], test[,-11], training[,11], k=7)

# Classifying based on EmployerState
Predict_h1b_2016 <- knn(training[,-3], test[,-3], training[,3], k=7)

#######################################################
#Error value for kNN when classifying with CASE_STATUS
#######################################################
e_result <- cbind(test, as.character(Predict_h1b_2016)) 
err_rate_2016 <- sum(e_result[,10]!=e_result[,11])/length(e_result[,10]!=e_result[,11])
err_rate_2016

accuracy_2016=(1-err_rate_2016)*100;
accuracy_2016

# Removing columns with zero variance 

H1B_RF<-H1B[,-c(2,4)]

H1B_2016<-H1B_RF[H1B_RF$CASE_SUBMITTED_YEAR==2016,]
H1B_2017<-H1B_RF[H1B_RF$CASE_SUBMITTED_YEAR==2017,]
H1B_2017<-H1B_2017[1:85945,]

set.seed(123)
index<-sort(sample(nrow(H1B_2016),round(.30*nrow(H1B_2016))))
training<-H1B_2016[-index,]
test<-H1B_2016[index,]
training<- na.omit(training)
test<- na.omit(test)
str(training)

# random forest with smote

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 10, 
                     verboseIter = FALSE,
                     sampling = "smote")

H1B_2016<- na.omit(H1B_2016)

set.seed(42)
model_rf_smote <- caret::train(CASE_STATUS ~ .,
                               data = training,
                               method = "rf",
                               preProcess = c("scale", "center"),
                               trControl = ctrl)


model_rf_smote$finalModel


# Random Forest

fit <- randomForest( CASE_STATUS~., data=training, importance=TRUE, ntree=100, na.action = na.omit)
importance(fit)
varImpPlot(fit)

test[,9]

Prediction <- predict(fit, test)
table(actual=test[,9],Prediction)

wrong<- (test[,9]!=Prediction )
error_rate<-sum(wrong)/length(wrong)
error_rate



# Cart

CART_class_1<-rpart(CASE_STATUS~CASE_SUBMITTED_YEAR+VISA_CLASS+EMPLOYER_STATE+EMPLOYER_STATE+SOC_NAME+
                      TOTAL_WORKERS+FULL_TIME_POSITION+H.1B_DEPENDENT+WILLFUL_VIOLATOR+WORKSITE_STATE,data=training,method="class",control =rpart.control(minsplit=2000,minbucket=1, cp=0))
printcp(CART_class_1) # display the results
#plotcp(CART_class_1) # visualize cross-validation results
summary(CART_class_1) # detailed summary of splits

# plot tree
rpart.plot(CART_class_1,main="2016")
prp(CART_class_1,main="2016")
#fancyRpartPlot(CART_class_1,cex=1)
#?prp
# calculating error rate
CART_predict_1<-predict(CART_class_1,test, type="class")

CART_wrong_1<-sum(test[,12]!=CART_predict_1)
CART_error_rate_1<-CART_wrong_1/length(test[,12])
CART_error_rate_1
accuracy<-(1-CART_error_rate_1)*100
accuracy



##################################  For 2017 ########################################################



H1B_2017<-H1B_imp[H1B_imp$CASE_SUBMITTED_YEAR==2017,]
H1B_2017<-H1B_2017[1:85945,]

################################################ Converting into Numeric  ##########################################




H1B_2017$EMPLOYER_STATE<-as.integer(H1B_2017$EMPLOYER_STATE)
H1B_2017$EMPLOYER_COUNTRY<-as.integer(H1B_2017$EMPLOYER_COUNTRY)
H1B_2017$SOC_NAME<-as.integer(H1B_2017$SOC_NAME)
H1B_2017$FULL_TIME_POSITION<-as.integer(H1B_2017$FULL_TIME_POSITION)
H1B_2017$H.1B_DEPENDENT<-as.integer(H1B_2017$H.1B_DEPENDENT)
H1B_2017$WORKSITE_STATE<-as.integer(H1B_2017$WORKSITE_STATE)
H1B_2017$CASE_STATUS<-as.integer(H1B_2017$CASE_STATUS)
H1B_2017$WILLFUL_VIOLATOR<-as.integer(H1B_2017$WILLFUL_VIOLATOR)



############################################# Training and Test ######################################################

set.seed(123)
index<-sort(sample(nrow(H1B_2017),round(.30*nrow(H1B_2017))))
training<-H1B_2017[-index,]
test<-H1B_2017[index,]
training<- na.omit(training)


# tried with model matrix as attributes type factor since encoding as numeric has impact on Neural net accuracy. But in the code shown only with numeric
#m_2017<-model.matrix(~CASE_STATUS+EMPLOYER_STATE+SOC_NAME+TOTAL_WORKERS+FULL_TIME_POSITION+H.1B_DEPENDENT+WILLFUL_VIOLATOR+WORKSITE_STATE+WAGE_YEARLY, data=training)


net_bc2_2017<- neuralnet(CASE_STATUS~EMPLOYER_STATE+SOC_NAME+TOTAL_WORKERS+FULL_TIME_POSITION+H.1B_DEPENDENT+WILLFUL_VIOLATOR+WORKSITE_STATE+WAGE_YEARLY,training,hidden=1,threshold=0.01)


#Plot the neural network
plot(net_bc2_2016)

str(test[,c(-1,-11)])

net_bc2_results <-compute(net_bc2_2016, test[,c(-1,-11)])
ANN=as.numeric(net_bc2_results$net.result)
ANN_round<-round(ANN)
ANN_cat<-ifelse(ANN<2.5,2,4)
table(Actual=test$Class,ANN_cat)
wrong<- (test$Class!=ANN_cat)
rate<-sum(wrong)/length(wrong)



# Using original dataset for KNn since column were removed with 

H1B$EMPLOYER_STATE<-as.numeric(H1B$EMPLOYER_STATE)
H1B$EMPLOYER_COUNTRY<-as.numeric(H1B$EMPLOYER_COUNTRY)
H1B$SOC_NAME<-as.numeric(H1B$SOC_NAME)
H1B$FULL_TIME_POSITION<-as.numeric(H1B$FULL_TIME_POSITION)
H1B$H.1B_DEPENDENT<-as.numeric(H1B$H.1B_DEPENDENT)
H1B$WORKSITE_STATE<-as.numeric(H1B$WORKSITE_STATE)
H1B$CASE_STATUS<-as.factor(H1B$CASE_STATUS)
H1B$WILLFUL_VIOLATOR<-as.numeric(H1B$WILLFUL_VIOLATOR)
H1B$VISA_CLASS<- as.numeric(H1B$VISA_CLASS)

H1B_2016<-H1B[H1B$CASE_SUBMITTED_YEAR==2016,]
H1B_2017<-H1B[H1B_$CASE_SUBMITTED_YEAR==2017,]
H1B_2017<-H1B_2017[1:85945,]

set.seed(123)
index<-sort(sample(nrow(H1B_2017),round(.30*nrow(H1B_2017))))
training<-H1B_2017[-index,]
test<-H1B_2017[index,]
training<- na.omit(training)
test<- na.omit(test)
str(training)

training[,3]

# Knn

# Classifying based on CASE_STATUS
Predict_h1b_2016 <- knn(training[,-11], test[,-11], training[,11], k=7)

# Classifying based on EmployerState
Predict_h1b_2016 <- knn(training[,-3], test[,-3], training[,3], k=7)

#######################################################
#Error value for kNN when classifying with CASE_STATUS
#######################################################
e_result <- cbind(test, as.character(Predict_h1b_2016)) 
err_rate_2016 <- sum(e_result[,10]!=e_result[,11])/length(e_result[,10]!=e_result[,11])
err_rate_2016

accuracy_2016=(1-err_rate_2016)*100;
accuracy_2016

# Removing columns with zero variance 

H1B_RF<-H1B[,-c(2,4)]

H1B_2017<-H1B[H1B_RF$CASE_SUBMITTED_YEAR==2017,]
H1B_2017<-H1B_2017[1:85945,]


set.seed(123)
index<-sort(sample(nrow(H1B_2017),round(.30*nrow(H1B_2017))))
training<-H1B_2017[-index,]
test<-H1B_2017[index,]
training<- na.omit(training)
test<- na.omit(test)
str(training)

# Random Forest

# With Smote
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 10, 
                     verboseIter = FALSE,
                     sampling = "smote")

H1B_2016<- na.omit(H1B_2016)

set.seed(42)
model_rf_smote <- caret::train(CASE_STATUS ~ .,
                               data = training,
                               method = "rf",
                               preProcess = c("scale", "center"),
                               trControl = ctrl)

fit <- randomForest( CASE_STATUS~., data=training, importance=TRUE, ntree=100, na.action = na.omit)
importance(fit)
varImpPlot(fit)

test[,9]

Prediction <- predict(fit, test)
table(actual=test[,9],Prediction)

wrong<- (test[,9]!=Prediction )
error_rate<-sum(wrong)/length(wrong)
error_rate



# Cart

CART_class_1<-rpart(CASE_STATUS~CASE_SUBMITTED_YEAR+VISA_CLASS+EMPLOYER_STATE+EMPLOYER_STATE+SOC_NAME+
                      TOTAL_WORKERS+FULL_TIME_POSITION+H.1B_DEPENDENT+WILLFUL_VIOLATOR+WORKSITE_STATE,data=training,method="class",control =rpart.control(minsplit=2000,minbucket=1, cp=0))
printcp(CART_class_1) # display the results
#plotcp(CART_class_1) # visualize cross-validation results
summary(CART_class_1) # detailed summary of splits

# plot tree
rpart.plot(CART_class_1,main="2016")
prp(CART_class_1,main="2016")
#fancyRpartPlot(CART_class_1,cex=1)
#?prp
# calculating error rate
CART_predict_1<-predict(CART_class_1,test, type="class")

CART_wrong_1<-sum(test[,12]!=CART_predict_1)
CART_error_rate_1<-CART_wrong_1/length(test[,12])
CART_error_rate_1
accuracy<-(1-CART_error_rate_1)*100
accuracy
