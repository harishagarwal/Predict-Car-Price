setwd("/Users/harishagarwal/Desktop/Harish/Data Science Courses/Git Projects/Predict Car price")

library(stringr)
library(tidyr)
library(ggplot2)
library(dplyr)
library(fastDummies)
library(knitr)

data = read.csv(file = "CarPrice_Assignment.csv", header = T, na.strings = c('',' ', NULL))

#Looking at the structure of data 
head(data)
summary(data)
str(data)

#Cleaning data
data$drivewheel[which(data$drivewheel == '4wd')] = 'fwd'
data$drivewheel = factor(data$drivewheel)
table(data$drivewheel)

#Since EngineLocation has most of them as front, we can remove this column from our analysis because it won't have any impact on the dependent variable
table(data$enginelocation)
data$enginelocation = NULL

#Split CarName to get the Car Brand, since we don't want to use the model name in our analysis
data = separate(data, col = CarName, into = c("CarBrandName", "CarModelName"), sep = " ")
data$CarModelName = NULL

summary(data)

#Since Symboling is the assigned insurance risk rating, let's convert it into a categorical variable
#A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.(Categorical)
data$symboling = as.factor(data$symboling)

#Creating a linear model to check for significant variables

model1 = lm(price ~ ., data = data)
summary(model1)

mod_init <- lm(price ~ ., data = data)
mod_best <- step(mod_init, direction = "both")
summary(mod_best)

#Let's look at the distribution of Symboling var. and also the variation of price with it
#Also, from the boxplot below, we can clearly see that the autos with lower risk ratings have higher prices

data %>% ggplot(aes(x = symboling)) + geom_bar()
data %>% ggplot(aes(x = symboling, y = price)) + geom_boxplot()

#Now let's see how the car brands are distributed across, and the price variation
#Let's account for the mis-spelled brand names and rename them first
table(data$CarBrandName)

data$CarBrandName[which(data$CarBrandName == 'maxda')] = 'mazda'
data$CarBrandName[which(data$CarBrandName == 'porcshce')] = 'porsche'
data$CarBrandName[which(data$CarBrandName == 'toyouta')] = 'toyota'
data$CarBrandName[which(data$CarBrandName == 'vokswagen')] = 'volkswagen'
data$CarBrandName[which(data$CarBrandName == 'vw')] = 'volkswagen'
data$CarBrandName[which(data$CarBrandName == 'Nissan')] = 'nissan'

table(data$CarBrandName)
data %>% ggplot(aes(x = CarBrandName, y = price, color = CarBrandName)) + geom_point()

data %>% group_by(CarBrandName) %>% summarise(avgPrice = mean(price)) %>% 
  ggplot(aes(x = CarBrandName, y = avgPrice, fill = CarBrandName)) + geom_bar(stat = "identity")

#Check if Car Brand is related with other var like engine size, engine type, carbody, cylinder number

#From the plot below, we can say that car brands with higher price also have bigger sized engines
data %>% ggplot(aes(x = CarBrandName, y = price, color = CarBrandName, size = enginesize)) + geom_point()

#Try dimensionality reduction for categorical variables to only get the significant variables

library(factoextra)
categorical_var = c('symboling', 'CarBrandName','fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem')
factored = data[colnames(data) %in% categorical_var]

#dummies <- as.data.frame(sapply(data_pca, function(x) data.frame(model.matrix(~x,data=data_pca)) [ , -1]))

dummies = dummy_cols(factored, remove_first_dummy = TRUE)
dummies = dummies[-c(1:length(categorical_var))] #Removing original columns, and keeping only the dummies
encoded_dataset = cbind(data %>% select(-categorical_var), dummies)
str(encoded_dataset)

encoded_dataset = encoded_dataset %>% rename("symboling_minus2" = "symboling_-2" , "symboling_minus1" = "symboling_-1")
res.pca <- prcomp(encoded_dataset, scale = TRUE)

#Eigen-values:
eig.val <- get_eigenvalue(res.pca)
eig.val
fviz_eig(res.pca, ncp = 50) #Scree plot to explain variation per dimension

#The above plot shows that based on the scree plot results, it seems that 35 dimensions explain ~95% of the variance in the dataset.
#By aggregating the individual variance contributions for the first 35 PCs, I will come up with a feature score to identify the top 35 features that explains the dataset:

library(tidyr)
library(glue)

#Results for Variables:
res.var <- get_pca_var(res.pca) # Variance Contributions of features to the PCs

pca_feature_selection = function(dim_size) {
  
  pca_feature_selection = data.frame(
    cbind(row.names(res.var$contrib), as.data.frame(res.var$contrib))) %>%
    rename(features = row.names.res.var.contrib.) %>%
    select(seq(1,dim_size + 1,1)) %>% #selecting required principal dimensions (45 cols here) from the data
    gather(key = "pc_dimension",
           value = "contribution",
           paste("Dim.", seq(dim_size), sep="") #converting wide to long format
    ) %>%
    mutate(features = as.character(features)) %>%
    group_by(features) %>%
    summarise(contribution = sum(contribution)) %>%
    arrange(desc(contribution)) %>%
    top_n(dim_size) %>%
    select(features) %>% t()
  
}

pca_feature_selection = pca_feature_selection(35)
as.data.frame(pca_feature_selection)

features_string = paste("price", paste(pca_feature_selection, collapse = '+'), sep = '~')
formula = as.formula(features_string)
formula

#Creating training and testing datasets
set.seed(123)
train_data = encoded_dataset %>% sample_frac(0.7)
test_data = encoded_dataset %>% anti_join(train_data, by = 'car_ID')

#Creating a linear regression model using :
#10-fold Cross Validation
#2 complete sets of folds to compute
#measuring the MAE to test for goodness of fit

library(caret)

#Creating a function which builds a linear regression model on train data

model_lm_train = function(dataset, formula, kfold_cv, cv_repeat, seed, minimize_metric){
  
  control = trainControl(method = 'repeatedcv', number = kfold_cv, repeats = cv_repeat)
  
  set.seed(seed)
  model = train(formula, data = dataset, method = 'lm', trControl = control, metric = minimize_metric)
  
  return(model)
}

model_lm = model_lm_train(train_data, formula, kfold_cv = 10, cv_repeat = 2, seed = 523, minimize_metric = 'MAE')
summary(model_lm)

lm_imp <- varImp(model_lm)
plot(lm_imp)

#Predict the Price for test dataset
prediction_lm = predict(model_lm, test_data)
eval_results_lm = postResample(prediction_lm, test_data$price)
eval_results_lm

RMSE_lm_test = eval_results_lm[1]
Rsquared_lm_test = eval_results_lm[2]

RMSE_lm_train = model_lm[[4]][2]
Rsquared_lm_train = model_lm[[4]][3]

#Calculate MAPE for training and testing datasets
MAPE_lm_test = mean(abs(prediction_lm - test_data$price)/test_data$price)

prediction_lm_train = predict(model_lm, train_data)
MAPE_lm_train = mean(abs(prediction_lm_train - train_data$price)/train_data$price)

#Find the average price of all cars across the dataset
avgPriceTotal = mean(data$price)
avgPriceTotal

as.data.frame(cbind(MAPE_lm_test, MAPE_lm_train, RMSE_lm_test, RMSE_lm_train, Rsquared_lm_test, Rsquared_lm_train))

#Random Forest Model

model_rf_train = function(dataset, formula, kfold_cv, cv_repeat, seed, minimize_metric){
  
  control = trainControl(method = "repeatedcv", number = kfold_cv, repeats = cv_repeat)
  tunegrid <- expand.grid(mtry = 4)
  set.seed(seed)
  
  model= train(formula, 
               data = dataset, 
               method = "rf", 
               tuneGrid = tunegrid, 
               trControl = control, 
               metric = minimize_metric,
               importance = TRUE
               )
  return(model)
}

model_rf = model_rf_train(train_data, formula, kfold_cv = 10, cv_repeat = 2, seed = 523, minimize_metric = "MAE")
summary(model_rf)

rf_imp = varImp(model_rf, scale = FALSE)
plot(rf_imp)

#Predict the Price for test dataset
prediction_rf = predict(model_rf, test_data)
eval_results_rf = postResample(prediction_rf, test_data$price)
eval_results_rf

RMSE_rf_test = eval_results_rf[1]
Rsquared_rf_test = eval_results_rf[2]

RMSE_rf_train = model_rf[[4]][2]
Rsquared_rf_train = model_rf[[4]][3]

#Calculate MAPE for training and testing datasets
MAPE_rf_test = mean(abs(prediction_rf - test_data$price)/test_data$price)

prediction_rf_train = predict(model_rf, train_data)
MAPE_rf_train = mean(abs(prediction_rf_train - train_data$price)/train_data$price)

as.data.frame(cbind(MAPE_rf_test, MAPE_rf_train, RMSE_rf_test, RMSE_rf_train, Rsquared_rf_test, Rsquared_rf_train))

# We can see that the RF model has a relatively better Rsquared than LM, but it has a higher MAPE and RMSE.

#XGBoost Model

model_xgb_train = function(dataset, formula, kfold_cv, cv_repeat, seed, minimize_metric){
  
  #Model Parameters:
  control <- trainControl(method="repeatedcv", 
                          number= kfold_cv,
                          search = "random",
                          repeats= cv_repeat
  )
  set.seed(seed)
  tunegrid <- expand.grid(nrounds = 728,
                          max_depth = 2,
                          eta = 0.4554021,
                          gamma = 2.74075,
                          colsample_bytree = 0.592694,
                          min_child_weight = 19,
                          subsample = 0.9610343)
  
  #Model Training:
  model <- train(formula, 
                 data=dataset, 
                 method="xgbTree", 
                 tuneGrid = tunegrid,
                 metric=minimize_metric,
                 trControl=control,
                 importance= TRUE
                 )
  return(model)
}

model_xgb = model_xgb_train(train_data, formula, kfold_cv = 10, cv_repeat = 2, seed = 523, minimize_metric = "MAE")
model_xgb
summary(model_xgb)


