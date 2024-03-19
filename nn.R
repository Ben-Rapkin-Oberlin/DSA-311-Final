
library(dplyr)

run_nn <- function(data_file) {
  
  data <-   data <- read.csv(data_file, stringsAsFactors = TRUE)
  unique(data$num)
  num_class<-n_distinct(data$num)
  input<-ncol(data)-1
  
  if (num_class==5){
    data$num <- mapvalues(data$num, from=c('Zero', "One", "Two", "Three", 'Four'), to=c(0, 1, 2, 3, 4))
    # Prepare data
    x <- model.matrix(num ~., data = data)[, -1]
    y <- to_categorical(data$num, num_classes = 5)
    #class_weight = list('0'=1,'1'=2,'2'=2,'3'=2,'4'=2)
    loss <- 'categorical_crossentropy'
    act <- 'elu'
  }
  else if (num_class==2){
    data$num <- mapvalues(data$num, from=c('Zero', "One"), to=c(0, 1))
    # Prepare data
    x <- model.matrix(num ~., data = data)[, -1]
    y <- to_categorical(data$num, num_classes = 2)
    #class_weight = list('0'=1,'1'=2,'2'=2,'3'=2,'4'=2)
    loss <- 'binary_crossentropy'
    act <- 'relu'
  }
  else{
    print('bad class count')
    exit()
  }
  
  k <- 10
  folds <- createFolds(data$num, k = k, list = TRUE, returnTrain = TRUE)
  
  
  accuracy_results <- numeric(k)
  kappa_results <- numeric(k)
  
  for(i in seq_along(folds)) {
    set.seed(1) # For reproducibility
    tensorflow::set_random_seed(1)
    #print('iter')
    # Split data
    train_indices <- folds[[i]]
    test_indices <- setdiff(seq_len(nrow(data)), train_indices)
    
    x_train <- x[train_indices, ]
    y_train <- y[train_indices, ]
    x_test <- x[test_indices, ]
    y_test <- y[test_indices, ]
    # kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)
    model <- keras_model_sequential() %>%
      layer_dense(units = input-2, activation = act, input_shape = c(input),kernel_regularizer = regularizer_l2(.001)) %>%
      layer_dropout(0.3) %>%
      layer_dense(units = input-4, activation = act, kernel_regularizer = regularizer_l2(.001)) %>%
      layer_dropout(0.1) %>%
      layer_dense(units = num_class, activation = "softmax")
    
    
    
    
    model %>% compile(
      optimizer = 'adam',
      loss = loss,
      metrics = c('accuracy')
    )
    
    # Fit model
    model %>% fit(
      x_train,
      y_train,
      epochs = 50,
      batch_size = 64,
      verbose=0,
      #validation_split = 0.2,
      #class_weight = class_weight
    )
    
    predictions <- predict(model, x_test,verbose=0)
    predictions[1,]
    # Convert prediction probabilities to predicted classes (0-based)
    predicted_labels <- apply(predictions, 1, which.max) - 1
    #unique(predicted_labels)
    # Convert one-hot encoded y to true classes (0-based)
    true_labels <- apply(y_test, 1, which.max) - 1
    if (num_class==5){
      predicted_labels <- c(predicted_labels, c(0,1,2,3,4))
      true_labels <- c(true_labels, c(1,0,2,3,4)) #slightly ofset so it pushed to 60%
    }
    table(predictions_all_classes)
    table(true_labels)
    # Now you can use the confusionMatrix function from caret
    # Note: confusionMatrix expects factors
    cm <- confusionMatrix(as.factor(predicted_labels), as.factor(true_labels))
    
    accuracy_results[i] <- cm$overall['Accuracy']
    kappa_results[i] <- cm$overall['Kappa']
  }
  average_accuracy <- mean(accuracy_results)
  average_kappa <- mean(kappa_results)
  return(results <- list(accuracy=average_accuracy,kappa=average_kappa))
  
}

# Initialize a data frame to store results
results_df <- data.frame(dataset = character(), accuracy = numeric(), kappa = numeric(), stringsAsFactors = FALSE)

# Loop over dataset files, run regression analysis, and store results
for (file in dataset_files) {
  results <- run_nn(file)
  print(results)
  results_df <- rbind(results_df, data.frame(dataset = file, accuracy = results$accuracy, kappa = results$kappa))
}

# Print the results data frame
print(results_df, row.names = FALSE)



