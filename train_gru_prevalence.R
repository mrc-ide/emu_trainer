library(torch)
library(luz)
library(dplyr)
library(ggplot2)
library(patchwork)

# Load functions
source("R/utils.R")

set.seed(123321)

# Data -------------------------------------------------------------------------
# Raw data
data_raw <- readRDS("data/data_large.RDS") 

transform_ccc <- sqrt_min_max_tf(data_raw$ccc)
transform_prevalence_2_10 <- sqrt_min_max_tf(data_raw$prevalence_2_10)

data <- data_raw |>
  mutate(id = paste(name, "_", eir)) |>
  #mutate(t = t - (5 * 365)) |>
  #filter(t > 0) |>
  mutate(
    ccc = transform_ccc$transform(ccc),
    prevalence_2_10 = transform_prevalence_2_10$transform(prevalence_2_10)
  )
# ------------------------------------------------------------------------------

# Torch data -------------------------------------------------------------------
data_array <- simplify2array(by(data[, c("ccc", "prevalence_2_10")], data$id, as.matrix)) |>
  aperm(c(3, 1, 2))

t_m <- dim(data_array)[2] # Max time
t_s <- 1 + (365 * 5) # Start time we want to evaluate prediction from
x <- torch_tensor(data_array[, , 1, drop = FALSE])
y <- torch_tensor(data_array[, t_s:t_m, 2, drop = FALSE])
ds <- tensor_dataset(x, y)

# Split into train (60%), validate (20%) and test (20%)
# names <- unique(data$name)
# train_names <- sample(names, size = 0.6 * length(names))
# valid_names <- sample(setdiff(names, train_names), size = 0.2 * length(names))
# test_names <- setdiff(names, union(train_names, valid_names))
# 
# ids <- unique(data$id)
# train_ids <- sapply(train_names, function(x){
#   which(grepl(x, ids))
# }) |>
#   as.vector()
# valid_ids <- sapply(valid_names, function(x){
#   which(grepl(x, ids))
# }) |>
#   as.vector()
# test_ids <- sapply(test_names, function(x){
#   which(grepl(x, ids))
# }) |>
#   as.vector()

eirs <- unique(data[,c("eir", "name")])$eir
p <- 1 / sqrt(eirs - (eirs / 20))
train_ids <- sample(1:length(ds), size = 0.8 * length(ds), prob = p)
table(eirs)
table(eirs[train_ids])

valid_ids <- sample(
  setdiff(1:length(ds), train_ids),
  size = 0.1 * length(ds)
)
test_ids <- setdiff(
  1:length(ds),
  union(train_ids, valid_ids)
)

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(
  train_ds,
  batch_size = 16,
  shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 16)
test_dl <- dataloader(test_ds, batch_size = length(test_ids))
# ------------------------------------------------------------------------------

# Specify & define the model ---------------------------------------------------
net <- nn_module(
  initialize = function(d_in, d_hidden, n_layers) {
    self$net <- nn_gru(
      input_size = d_in,
      hidden_size = d_hidden,
      num_layers = n_layers,
      batch_first = TRUE
    )
    self$dropout <- nn_dropout(p = 0.2)
    self$output <- nn_linear(d_hidden, 1)
  },
  forward = function(x) {
    (x |>
       self$net())[[1]][, 1826:dim(x)[2], ] |>
      self$dropout() |>
      self$output()  |>
      nnf_relu()
  }
)

# Define the model
model <- net |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_mae()
      )
  )|>
  set_hparams(
    d_in = 1,
    d_hidden = 15,
    n_layers = 1
  )  |>
  set_opt_hparams(
    lr = 0.1
  )
# ------------------------------------------------------------------------------

# Learning rate finder ---------------------------------------------------------
learning_rate <- FALSE
if(learning_rate){
  # Learning rate finder
  rates_and_losses <- model |>
    lr_finder(
      train_dl,
      start_lr = 0.001,
      end_lr = 5
    )
  rates_and_losses |>
    plot()
}
# ------------------------------------------------------------------------------

# Fit the model ----------------------------------------------------------------
num_epochs <- 200
fitted <- model |>
  fit(
    train_dl,
    epochs = num_epochs,
    valid_data = valid_dl,
    callbacks = list(
      luz_callback_lr_scheduler(
        lr_step,
        step_size = 10
      ),
      luz_callback_early_stopping(
        patience = 5
      )
    )
  )
# ------------------------------------------------------------------------------

# Evaluate the model -----------------------------------------------------------
eval <- fitted |>
  evaluate(test_dl)
eval

metrics <- fitted |>
  get_metrics()
ggplot(metrics, aes(x = epoch, y = value, col = set)) + 
  geom_line() + 
  scale_y_log10() +
  theme_bw() +
  facet_wrap(~metric)
# ------------------------------------------------------------------------------

# Performance for test set -----------------------------------------------------
pred <- fitted |>
  predict(test_dl)

# Format
prevalence_2_10_predict <- as.vector(t(as.array(pred[,,1])))
prevalence_2_10 <- as.vector(t(as.array(test_ds$dataset$tensors[[2]][test_ds$indices,][,,1])))
t_p <- seq_along(t_s:t_m)
pd <- data.frame(
  prevalence_2_10_predict = prevalence_2_10_predict,
  prevalence_2_10 = prevalence_2_10,
  t = t_p,
  group = rep(1:(length(prevalence_2_10) / length(t_p)), each = length(t_p)),
  id = rep(unique(data$id)[test_ds$indices], each = length(t_p))) |>
  tidyr::separate_wider_delim(
    cols = "id",
    delim = " _ ",
    names = c("name", "eir"),
    cols_remove = FALSE
  ) |>
  mutate(id = factor(id),
         id = forcats::fct_reorder(id, eir)) |>
  # Untransform
  mutate(
    prevalence_2_10_predict = transform_prevalence_2_10$untransform(prevalence_2_10_predict), 
    prevalence_2_10 = transform_prevalence_2_10$untransform(prevalence_2_10)
  )

test_correlation_plot <- ggplot(
  data = slice_sample(pd, n = 5000),
  aes(x = prevalence_2_10, y = prevalence_2_10_predict)) +
  geom_point(alpha = 0.2) + 
  geom_abline(intercept = 0, slope = 1, col = "deeppink") +
  theme_bw()
ggsave("figures/gru_prevalence_correlation_plot.pdf", test_correlation_plot)

test_comparison_plot <- ggplot(data = pd) +
  geom_line(aes(x = t, y = prevalence_2_10)) +
  geom_line(aes(x = t, y = prevalence_2_10_predict), col = "deeppink") +
  theme_bw() +
  #ylim(0, 1) +
  facet_wrap(~ id, scales = "free_y")

test_fit_plots <- list()
ids <- unique(pd$group)
for(i in seq_along(ids)){
  td <- filter(pd, group == i)
  p <- ggplot(data = td) +
    geom_line(aes(x = t, y = prevalence_2_10)) +
    geom_line(aes(x = t, y = prevalence_2_10_predict), col = "deeppink") +
    theme_bw() +
    ggtitle(ids[i])
  test_fit_plots[[i]] <- p
}

pdf("figures/gru_prevalence_timeseries_plot.pdf", onefile = TRUE)
i <- 1
while(i < length(test_fit_plots)) {
  print(wrap_plots(test_fit_plots[i:(i + 5)], ncol = 2))
  i <- i + 6
}
dev.off()
# ------------------------------------------------------------------------------

# Save the model
save_model <- FALSE
if(save_model){
  luz_save(fitted, "models/gru_sqrt_prevalence_hidden15_relu.rds")
}