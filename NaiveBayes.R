#library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(ggplot2)  # data visualization
library(caret)    # implementing with caret
#install.packages('h2o')
library(h2o)      # implementing with h2o
#h2o.no_progress()
#h2o.init()

# Function to load data, run regression analysis, and return accuracy and Kappa
run_regression_analysis <- function(data_file) {
  # Set seed for reproducibility
  set.seed(1)
  
  # Load dataset
  data <- read.csv(data_file, stringsAsFactors = TRUE)
  
  # Create the model matrix
  x <- model.matrix(num ~., data = data)[, -1]
  y <- data$num
  
  # Set up k-fold cross-validation
  train_control <- trainControl(
    method = "cv", 
    number = 10
  )
  
  #nb_grid <- expand.grid(usekernel = c(TRUE, FALSE),
  #                       laplace = c(0, 0.5, 1), 
  #                       adjust = c(0.75, 1, 1.25, 1.5))
  
  # Fit the Naive Bayes model with parameter tuning
  set.seed(2550)
  naive_bayes_via_caret2 <- train(x, y, method = "multinom", trControl = train_control)#,tuneGrid=nb_grid)
  
  # View the selected tuning parameters
  print(naive_bayes_via_caret2$finalModel$tuneValue)
  
  
  # Make predictions on the entire dataset
  predictions <- predict(naive_bayes_via_caret2$finalModel, newdata = data, mode = "everything",)
  
  # Confusion Matrix
  confusion_matrix <- confusionMatrix(predictions, data$num)
  #print(confusion_matrix)
  # Extract accuracy and Kappa
  results <- list(accuracy = confusion_matrix$overall['Accuracy'], kappa = confusion_matrix$overall['Kappa'])
  return(results)
}

# List of dataset files
dataset_files <- c("Encoded_Scaled.csv", "pca_data.csv", "Simple_Encoded_Scaled.csv", "simple_pca_data.csv")

# Initialize a data frame to store results
results_df <- data.frame(dataset = character(), accuracy = numeric(), kappa = numeric(), stringsAsFactors = FALSE)

# Loop over dataset files, run regression analysis, and store results
for (file in dataset_files) {
  results <- run_regression_analysis(file)
  results_df <- rbind(results_df, data.frame(dataset = file, accuracy = results$accuracy, kappa = results$kappa))
}

# Print the results data frame
print(results_df, row.names = FALSE)





