library(brms)
library(dplyr)
library(lubridate)
library(tidyverse)
library(cmdstanr)
options(brms.backend = "cmdstanr")
options(mc.cores = 4)

track_df = read_csv("./Data/track_df.csv")

fs_gamma <- bf(
  track_90q_count ~
    s(air.2m.night) +
    s(air.2m.night.diff) +
    s(wind_speed.night, k = 12) +
    s(prate.night)+
    s(rhum_resid) +
    s(pres.sfc.night) +
    s(tcdc.night) +
    s(lunar.illumination, k = 15) +
    s(avg_drought_index) +
    roost_type +
    s(dist) +
    s(from_sunset) +
    s(avg_3day_prior) +
    s(yday, bs = "cc") +
    s(yday, colony, bs = "fs", xt = list(bs = "cc"), k = 8) + # factor smooth: gives each colony its own seasonal emergence curve around the population mean — this alone often soaks up a lot of what looks like autocorrelation, because colonies differ systematically in timing, not just in level.
    (1 | colony) +
    (1 | year) 
)

# Check what priors brms expects for this formula before fitting
default_prior(fs_gamma, data = track_df, family = Gamma(link = "log"))

priors <- c(
  prior(normal(0, 5), class = "b"),               # fixed effects (roost_type)
  prior(student_t(3, 0, 2.5), class = "sds"),     # smooth term wiggliness (all s() terms)
  prior(student_t(3, 0, 2.5), class = "sd"),      # colony/year random intercept SDs
  prior(normal(0, 0.3), class = "ar"),            # AR(1) coefficient
  prior(gamma(0.01, 0.01), class = "shape")       # Gamma shape parameter (replaces sigma)
)

fit <- brm(
  fs_gamma,
  data = track_df,
  Gamma(link = "log"),
  prior = priors,
  chains = 4,
  cores = 4,
  iter = 2000,
  warmup = 1000,
  backend = "cmdstanr",     # substantially faster than rstan for this size
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  seed = 123,
  output_dir = "./Models/csvs"  # <-- Saves raw chain CSVs here permanently
)

saveRDS(fit, "./Models/fs_gamma.rds")
