# Supplementary Table S14 (step 2): minimum sample size per model (Riley et al., pmsampsize).
# Run after sample_size.py. Requires: install.packages("pmsampsize")
library(pmsampsize)

out_dir <- Sys.getenv("T2D_OUT_DIR", "../results")
d <- read.csv(file.path(out_dir, "ss_input.csv"), fileEncoding = "UTF-8")

res <- do.call(rbind, lapply(seq_len(nrow(d)), function(i) {
  r <- d[i, ]
  o <- tryCatch(
    pmsampsize(type = "b", cstatistic = r$cstat, parameters = r$params,
               prevalence = r$prev, shrinkage = 0.9),
    error = function(e) NULL)
  if (is.null(o)) return(NULL)
  data.frame(stratum = r$stratum, available_n = r$n, available_events = r$events,
             required_n = o$sample_size, required_events = round(o$events),
             required_EPP = o$EPP, R2_CS = round(o$rsquared, 4))
}))

res$shortfall_n <- res$required_n - res$available_n
res$adequate    <- ifelse(res$shortfall_n <= 0, "yes", "NO")
write.csv(res, file.path(out_dir, "ss_results.csv"), row.names = FALSE)
print(res, row.names = FALSE)
cat("\nwrote", file.path(out_dir, "ss_results.csv"), "\n")
