library(mgcv)
library(dplyr)
library(lubridate)
library(tidyverse)
library(sf)
library(ggplot2)
library(maps)
library(dplyr)
library(brms)
library(Metrics) # for rmse, mae

track_df = read_csv("./Data/track_df.csv")
fit_full <- readRDS("./Models/fs_ar.rds")

# 1. Create a blocked split (e.g., hold out 20% of series_ids)
set.seed(123)
all_series <- unique(track_df$series_id)
test_series <- sample(all_series, size = length(all_series) * 0.2)

train_df <- track_df |> filter(!series_id %in% test_series)
test_df  <- track_df |> filter(series_id %in% test_series)

# 2. Refit the model on the training data 
# (using update() is faster as it avoids recompiling the Stan code)
fit_train <- update(fit_full, newdata = train_df, cores = 4)
saveRDS(fit_train, "./Models/fit_full_train.rds")

# 3. Predict on the hold-out test set
# allow_new_levels = TRUE is necessary if any colonies in the test set weren't in the train set
preds <- predict(fit_train, newdata = test_df, allow_new_levels = TRUE)

# predict() returns a matrix with Estimate, Est.Error, Q2.5, Q97.5
test_df$pred_log_count <- preds[, "Estimate"]

# 4. Exponentiate back to original scale for ML evaluation
test_df <- test_df |> 
  mutate(
    pred_count = exp(pred_log_count),
    obs_count = track_90q_count
  )

# 5. Calculate ML-style metrics
rmse(test_df$obs_count, test_df$pred_count)
mae(test_df$obs_count, test_df$pred_count)