# Normalise
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Min max transform factory
min_max_tf <- function(x){
  min_x <- min(x)
  max_x <- max(x)
  transform <- function(x, min = min_x, max = max_x){
    (x - min) / (max - min)
  }
  untransform <- function(y, min = min_x, max = max_x){
    y * (max - min) + min
  }
  list(
    transform = transform,
    untransform = untransform
  )
}

log_min_max_tf <- function(x){
  t <- log(x + 1)
  min_x <- min(t)
  max_x <- max(t)
  transform <- function(x, min = min_x, max = max_x){
    x <- log(x + 1)
    (x - min) / (max - min)
  }
  untransform <- function(y, min = min_x, max = max_x){
    exp(y * (max - min) + min) - 1
  }
  list(
    transform = transform,
    untransform = untransform
  )
}

sqrt_min_max_tf <- function(x){
  t <- sqrt(x)
  min_x <- min(t)
  max_x <- max(t)
  transform <- function(x, min = min_x, max = max_x){
    x <- sqrt(x)
    (x - min) / (max - min)
  }
  untransform <- function(y, min = min_x, max = max_x){
    (y * (max - min) + min) ^ 2
  }
  list(
    transform = transform,
    untransform = untransform
  )
}

# Standardise
standardise <- function(x){
  (x - mean(x)) / sd(x)
}