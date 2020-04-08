

set.seed(123)

#Loading Libraries
library(keras)
library(reticulate)
library(tidyverse)


#INPUTTING DATA
n_sample <- 5000; maxlen <- 300; max_features <- 5000
imdb = readRDS("imdb.rds")
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb # Loads the data
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
sample_indicators = sample(1:nrow(x_train), n_sample)
x_train <- x_train[sample_indicators,] # use a subset of reviews for training
y_train <- y_train[sample_indicators] # use a subset of reviews for training
x_test <- x_test[sample_indicators,] # use a subset of reviews for testing
y_test <- y_test[sample_indicators] # use a subset of reviews for testing

#WRITING DATA TO RDS FILES
write_rds(x_test, "x_test.rds")
write_rds(y_test, "y_test.rds" )






#SIMPLE RNN

#MODEL CREATION##############
model_RNN <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 16) %>%  
  layer_simple_rnn(units = 16) %>%  
  layer_dense(units = 1, activation = "sigmoid")



#MODEL COMPILE##############

model_RNN %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


#######MODEL HISTORY###############

history_RNN <- model_RNN %>%  fit(
  x_train, y_train,
  epochs = 10, 
  batch_size = 128,
  validation_split = 0.3
)


##SAVING HISTORY FILES#########
model_RNN %>% save_model_hdf5("simplernn_model.h5")
write_rds(history_RNN, "simplernn_history.rds")

##################################################################################################


#LSTM

#MODEL CREATION##############
model_LSTM <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>%  
  layer_lstm(units = 32) %>%  
  layer_dense(units = 1, activation = "sigmoid")


#MODEL COMPILE##############
model_LSTM %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

#######MODEL HISTORY###############

history_LSTM <- model_LSTM %>%  fit(
  x_train, y_train,
  epochs = 10, 
  batch_size = 128,
  validation_split = 0.3
)


##SAVING HISTORY FILES#########
model_LSTM %>% save_model_hdf5("lstm_model.h5")
write_rds(history_LSTM, "lstm_history.rds")






############################################################################################



#GRU

#MODEL CREATION##############
model_GRU <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 16) %>%  
  layer_gru(units = 16) %>%  
  layer_dense(units = 1, activation = "sigmoid")

#MODEL COMPILE##############

model_GRU %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

#######MODEL HISTORY###############

history_GRU <- model_GRU %>%  fit(
  x_train, y_train,
  epochs = 12, 
  batch_size = 125,
  validation_split = 0.2
)
##SAVING HISTORY FILES#########
model_GRU %>% save_model_hdf5("gru_model.h5")
write_rds(history_GRU, "gru_history.rds")




###########################################################################################################




#Bidirectional lstm

#MODEL CREATION##############
model_BiLSTM <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 30) %>%  
  bidirectional(
    layer_lstm(units = 30)
  )%>%  
  layer_dense(units = 1, activation = "sigmoid")

#MODEL COMPILE##############
model_BiLSTM %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

#######MODEL HISTORY###############

history_BiLSTM <- model_BiLSTM %>%  fit(
  x_train, y_train,
  epochs = 10, 
  batch_size = 128,
  validation_split = 0.5
)




##SAVING HISTORY FILES#########
model_BiLSTM %>% save_model_hdf5("bilstm_model.h5")
write_rds(history_BiLSTM, "bilstm_history.rds")




###########################################################################################################



#Bidirectional gru

#MODEL CREATION##############

model_BiGRU <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 16) %>%  
  bidirectional(
    layer_gru(units = 16)
  )%>%  
  layer_dense(units = 1, activation = "sigmoid")

#MODEL COMPILE##############
model_BiGRU %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


#######MODEL HISTORY###############

history_BiGRU <-model_BiGRU  %>%  fit(
  x_train, y_train,
  epochs = 10, 
  batch_size = 140,
  validation_split = 0.5
)




##SAVING HISTORY FILES#########
model_BiGRU %>% save_model_hdf5("bigru_model.h5")
write_rds(history_BiGRU, "bigru_history.rds")




###########################################################################################################


#1d convnet

#MODEL CREATION##############
model_1dconv <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1)


#MODEL COMPILE##############
model_1dconv %>% compile(
    optimizer = optimizer_rmsprop(lr = 1e-4),
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

#######MODEL HISTORY###############

  
history_1dconv <- model_1dconv %>% fit(
    x_train, y_train,
    epochs = 15,
    batch_size = 160,
    validation_split = 0.6
  )
  



##SAVING HISTORY FILES#########
model_1dconv %>% save_model_hdf5("onedconv_model.h5")
write_rds(history_1dconv, "onedconv_history.rds")

