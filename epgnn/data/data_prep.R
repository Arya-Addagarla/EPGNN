if (!require("dplyr")) {
  install.packages("dplyr", repos = "http://cran.us.r-project.org")
  library(dplyr)
}
if (!require("readr")) {
  install.packages("readr", repos = "http://cran.us.r-project.org")
  library(readr)
}

eq_file <- "stead_earthquake.csv"
noise_file <- "stead_noise.csv"

if (!file.exists(eq_file) || !file.exists(noise_file)) {
  quit(status = 1)
}

eq_data <- read_csv(eq_file, show_col_types = FALSE)
noise_data <- read_csv(noise_file, show_col_types = FALSE)

eq_data <- eq_data %>% mutate(label = 1)
noise_data <- noise_data %>% mutate(label = 0)

combined_data <- bind_rows(eq_data, noise_data)

cleaned_data <- combined_data %>%
  filter(!is.na(receiver_latitude) & !is.na(receiver_longitude) & !is.na(trace_name)) %>%
  mutate(source_magnitude = ifelse(is.na(source_magnitude), 0, source_magnitude))

output_file <- "metadata_clean.csv"
write_csv(cleaned_data, output_file)
