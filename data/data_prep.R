# Install + load packages safely
if (!require("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cran.r-project.org")
}
library(dplyr)

if (!require("readr", quietly = TRUE)) {
  install.packages("readr", repos = "https://cran.r-project.org")
}
library(readr)

# File paths
eq_file <- "stead_earthquake.csv"
noise_file <- "stead_noise.csv"

# Check files exist
if (!file.exists(eq_file) || !file.exists(noise_file)) {
  stop("Missing input files")
}

# Read data
eq_data <- read_csv(eq_file, show_col_types = FALSE)
noise_data <- read_csv(noise_file, show_col_types = FALSE)

# Add labels
eq_data <- eq_data %>% mutate(label = 1)
noise_data <- noise_data %>% mutate(label = 0)

# Combine
combined_data <- bind_rows(eq_data, noise_data)

# Clean
cleaned_data <- combined_data %>%
  filter(
    !is.na(receiver_latitude),
    !is.na(receiver_longitude),
    !is.na(trace_name)
  ) %>%
  mutate(
    source_magnitude = ifelse(is.na(source_magnitude), 0, source_magnitude)
  )

# Output
output_file <- "metadata_clean.csv"
write_csv(cleaned_data, output_file)

cat("Saved to", output_file, "\n")
