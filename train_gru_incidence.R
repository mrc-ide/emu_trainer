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
transform_clinical <- sqrt_min_max_tf(data_raw$clinical)

data <- data_raw |>
  mutate(id = paste(name, "_", eir)) |>
  mutate(t = t - (5 * 365)) |>
  filter(t > 0) |>
  mutate(
    ccc = transform_ccc$transform(ccc),
    clinical = transform_clinical$transform(clinical)
  )
# ------------------------------------------------------------------------------

# Torch data -------------------------------------------------------------------
data_array <- simplify2array(by(data[, c("ccc", "clinical")], data$id, as.matrix)) |>
  aperm(c(3, 1, 2))

t_m <- dim(data_array)[2] # Max time
t_s <- 1 + (365 * 2) # Start time we want to evaluate prediction from
x <- torch_tensor(data_array[, , 1, drop = FALSE])
y <- torch_tensor(data_array[, t_s:t_m, 2, drop = FALSE])
ds <- tensor_dataset(x, y)

# Split into train (60%), validate (20%) and test (20%)
train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(
  setdiff(1:length(ds), train_ids),
  size = 0.2 * length(ds)
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
  batch_size = t_m,
  shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = t_m)
test_dl <- dataloader(test_ds, batch_size = t_m)
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
    self$dropout <- nn_dropout(p = 0.1)
    self$output <- nn_linear(d_hidden, 1)
    self$lr <- nn_leaky_relu(negative_slope = 0.01)
  },
  forward = function(x) {
    (x |>
       self$net())[[1]][, 731:dim(x)[2], ] |>
      self$dropout() |>
      self$output()  |>
      self$lr()
    #nnf_sigmoid()
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
    d_hidden = 10,
    n_layers = 1
  )  |>
  set_opt_hparams(
    lr = 0.01
  )
# ------------------------------------------------------------------------------

# Learning rate finder ---------------------------------------------------------
learning_rate <- FALSE
if(learning_rate){
  # Learning rate finder
  rates_and_losses <- model |>
    lr_finder(
      train_dl,
      start_lr = 0.0001,
      end_lr = 0.5
    )
  rates_and_losses |>
    plot()
}
# ------------------------------------------------------------------------------

# Fit the model ----------------------------------------------------------------
num_epochs <- 2000
fitted <- model |>
  fit(
    train_dl,
    epochs = num_epochs,
    valid_data = valid_dl,
    callbacks = list(
      luz_callback_lr_scheduler(
        lr_reduce_on_plateau,
        call_on = "on_batch_end"
      ),
      luz_callback_early_stopping(
        patience = 50
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
clinical_predict <- as.vector(t(as.array(pred[,,1])))
clinical <- as.vector(t(as.array(test_ds$dataset$tensors[[2]][test_ds$indices,][,,1])))
t_p <- seq_along(t_s:t_m)
pd <- data.frame(
  clinical_predict = clinical_predict,
  clinical = clinical,
  t = t_p,
  group = rep(1:(length(clinical) / length(t_p)), each = length(t_p)),
  id = rep(unique(data$id)[test_ds$indices], each = length(t_p))) |>
  tidyr::separate_wider_delim(
    cols = "id",
    delim = " _ ",
    names = c("name", "eir"),
    cols_remove = FALSE
  ) |>
  # Untransform
  mutate(
    clinical_predict = transform_clinical$untransform(clinical_predict), 
    clinical = transform_clinical$untransform(clinical)
  )

test_correlation_plot <- ggplot(
  data = slice_sample(pd, n = 5000),
  aes(x = clinical, y = clinical_predict)) +
  geom_point(alpha = 0.2) + 
  geom_abline(intercept = 0, slope = 1, col = "deeppink") +
  theme_bw()
ggsave("figures/gru_incidence_correlation_plot.pdf", test_correlation_plot)

test_comparison_plot <- ggplot(data = pd) +
  geom_line(aes(x = t, y = clinical)) +
  geom_line(aes(x = t, y = clinical_predict), col = "deeppink") +
  theme_bw() +
  facet_wrap(~ id, scales = "free_y")

test_fit_plots <- list()
ids <- unique(pd$group)
for(i in seq_along(ids)){
  td <- filter(pd, group == i)
  p <- ggplot(data = td) +
    geom_line(aes(x = t, y = clinical)) +
    geom_line(aes(x = t, y = clinical_predict), col = "deeppink") +
    theme_bw() +
    ggtitle(ids[i])
  test_fit_plots[[i]] <- p
}

pdf("figures/gru_incidence_timeseries_plot.pdf", onefile = TRUE)
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
  luz_save(fitted, "models/gru_sqrt_incidence_hidden15_leakyrelu0.01.rds")
}