library(future)
library(loo)
library(brms)
library(dplyr)
library(lubridate)
library(tidyverse)
library(cmdstanr)
options(mc.cores = 4)

track_df = read_csv("./Data/track_df.csv")
fit_full = readRDS("./Models/fs_ar.rds")

library(brms)
library(dplyr)

# --- Fix 1: convert local_date to numeric BEFORE fitting, avoids Date-class
# metadata mismatches in ar() validation entirely ---
track_df <- track_df |>
  mutate(local_date_num = as.numeric(local_date))  # days since epoch, numeric

# Refit your candidate models using local_date_num in ar(), e.g.:
# ar(time = local_date_num, gr = series_id, p = 1)
# (If you don't want to refit fit_full/fit_fs_ar_interact from scratch, you can 
#  likely get away with just using local_date_num for the CV refits below, since 
#  what matters for THIS check is that newdata construction stays consistent.)

# --- Manual grouped K-fold ---
set.seed(123)
series_ids <- unique(track_df$series_id)
K <- 5   # reduced from 10 — cuts total compute roughly in half; consider even 
# fewer folds given the runtime you're seeing
folds <- sample(rep(1:K, length.out = length(series_ids)))
fold_map <- setNames(folds, series_ids)
track_df$fold <- fold_map[as.character(track_df$series_id)]

# Storage for results
cv_results <- vector("list", K)

for (k in 1:K) {
  cat("=== Fold", k, "of", K, "===\n")
  
  train_k <- track_df |> filter(fold != k)
  test_k  <- track_df |> filter(fold == k)
  
  # Refit on training fold — reuse your model's formula/priors/family
  fit_k <- update(
    fit_full,                     # or fit_fs_ar_interact, whichever you're comparing
    newdata = train_k,
    chains = 1,
    cores = 1,
    iter = 2000,
    warmup = 1000,
    control = list(adapt_delta = 0.99, max_treedepth = 12),  # bumped up given divergence issue
    seed = 123
  )
  
  # Check divergences for THIS fold explicitly — don't silently trust it
  div <- rstan::get_num_divergent(fit_k$fit)  # or check via nuts_params(fit_k) if cmdstanr backend
  cat("Divergent transitions:", div, "\n")
  
  # Predict on held-out fold.
  # IMPORTANT: held-out series have NO history in train_k, so ar() has nothing 
  # to condition on for them — brms will effectively treat them as fresh series 
  # (AR contribution ~0 at the start of an unseen group), which is the expected/
  # correct behavior for genuinely out-of-sample series, not a bug to fix.
  pred_k <- posterior_predict(fit_k, newdata = test_k, allow_new_levels = TRUE)
  
  # allow_new_levels=TRUE needed since test_k's colonies weren't necessarily
  # all present in train_k depending on how series_id relates to colony —
  # double check: are colonies split across folds, or fully held out? See note below.
  
  pred_mean <- colMeans(pred_k)
  rmse_k <- sqrt(mean((pred_mean - log(test_k$track_90q_count))^2))
  
  cv_results[[k]] <- list(fit = fit_k, rmse = rmse_k, divergences = div)
  
  saveRDS(fit_k, paste0("./Models/fit_full_cv_fold_", k, ".rds"))  # checkpoint each fold
  saveRDS(cv_results, "./Models/cv_results.rds")
}

# Summarize
rmse_all <- sapply(cv_results, function(x) x$rmse)
div_all  <- sapply(cv_results, function(x) x$divergences)

data.frame(fold = 1:K, rmse = rmse_all, divergences = div_all)
mean(rmse_all)