# Generate compact R TrialEmulation reference outputs for the Python port.
#
# Package: TrialEmulation 0.0.4.9 (git f432103), installed from source into
# ~/R/te-ref. Data: the package's public example datasets `trial_example`
# and `data_censored` (simulated, not patient data).
#
# Usage: Rscript make_r_reference.R <out_dir>

.libPaths(c(path.expand("~/R/te-ref"), .libPaths()))
suppressPackageStartupMessages({
  library(TrialEmulation)
  library(data.table)
  library(sandwich)
})
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else "r_reference"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data("trial_example", package = "TrialEmulation")
data("data_censored", package = "TrialEmulation")

# Note: R 0.0.4.9 select_data_cols() collects cense_n_cov twice and never
# cense_d_cov, so every cense_d_cov variable must also appear in another
# formula. The data_censored configurations therefore carry all censoring
# covariates in the expansion outcome_cov; the MSM uses ~ x1 + x2 with linear
# trial-period and follow-up terms (the quadratic defaults are quasi-separated
# with 14 events).
configs <- list(
  te_itt = list(
    data = trial_example,
    prep = list(estimand_type = "ITT", outcome_cov = ~ nvarA + nvarB),
    msm = list(estimand_type = "ITT", outcome_cov = ~ nvarA + nvarB)
  ),
  te_pp = list(
    data = trial_example,
    prep = list(
      estimand_type = "PP", outcome_cov = ~ nvarA + nvarB,
      switch_n_cov = ~ nvarA + nvarB, switch_d_cov = ~ nvarA + nvarB + nvarC + catvarA
    ),
    msm = list(estimand_type = "PP", outcome_cov = ~ nvarA + nvarB)
  ),
  te_at = list(
    data = trial_example,
    prep = list(
      estimand_type = "As-Treated", outcome_cov = ~ nvarA + nvarB,
      switch_n_cov = ~ nvarA + nvarB, switch_d_cov = ~ nvarA + nvarB + nvarC + catvarA
    ),
    msm = list(estimand_type = "As-Treated", outcome_cov = ~ nvarA + nvarB)
  ),
  dc_itt_cens = list(
    data = data_censored,
    prep = list(
      estimand_type = "ITT", outcome_cov = ~ x1 + x2 + x3 + x4 + age_s,
      use_censor_weights = TRUE, cense = "censored", pool_cense = "numerator",
      cense_d_cov = ~ x1 + x2 + x3 + x4 + age_s, cense_n_cov = ~ x3 + x4
    ),
    msm = list(estimand_type = "ITT", outcome_cov = ~ x1 + x2, include_followup_time = ~followup_time, include_trial_period = ~trial_period)
  ),
  dc_pp_cens = list(
    data = data_censored,
    prep = list(
      estimand_type = "PP", outcome_cov = ~ x1 + x2 + x3 + x4 + age_s,
      switch_n_cov = ~ x1, switch_d_cov = ~ x1 + x2 + x3 + age_s,
      use_censor_weights = TRUE, cense = "censored", pool_cense = "none",
      cense_d_cov = ~ x1 + x2 + x3 + x4 + age_s, cense_n_cov = ~ x3 + x4
    ),
    # Only two events remain under PP censoring in this small example, so the
    # pooled logistic MSM is quasi-separated; no MSM reference is generated.
    msm = NULL
  ),
  dc_at_cens = list(
    data = data_censored,
    prep = list(
      estimand_type = "As-Treated", outcome_cov = ~ x1 + x2 + x3 + x4 + age_s,
      switch_n_cov = ~ x1, switch_d_cov = ~ x1 + x2 + x3 + age_s,
      use_censor_weights = TRUE, cense = "censored", pool_cense = "both",
      cense_d_cov = ~ x1 + x2 + x3 + x4 + age_s, cense_n_cov = ~ x3 + x4
    ),
    msm = list(estimand_type = "As-Treated", outcome_cov = ~ x1 + x2, include_followup_time = ~followup_time, include_trial_period = ~trial_period)
  )
)

summary_rows <- list()
for (name in names(configs)) {
  cfg <- configs[[name]]
  message("== ", name)
  prep <- do.call(
    data_preparation,
    c(
      list(
        data = cfg$data, id = "id", period = "period", treatment = "treatment",
        outcome = "outcome", eligible = "eligible", quiet = TRUE
      ),
      cfg$prep
    )
  )
  d <- as.data.table(prep$data)
  setorder(d, id, trial_period, followup_time)
  n <- nrow(d)

  # Expanded-data summary: one row per column.
  num_cols <- names(d)[vapply(d, is.numeric, logical(1))]
  col_summary <- data.table(
    config = name,
    column = names(d),
    n_rows = n,
    n_na = vapply(d, function(x) sum(is.na(x)), integer(1)),
    sum = vapply(names(d), function(cn) if (cn %in% num_cols) sum(as.double(d[[cn]]), na.rm = TRUE) else NA_real_, numeric(1)),
    min = vapply(names(d), function(cn) if (cn %in% num_cols) min(as.double(d[[cn]]), na.rm = TRUE) else NA_real_, numeric(1)),
    max = vapply(names(d), function(cn) if (cn %in% num_cols) max(as.double(d[[cn]]), na.rm = TRUE) else NA_real_, numeric(1))
  )
  q <- quantile(d$weight, c(0, 0.01, 0.25, 0.5, 0.75, 0.99, 1), na.rm = TRUE, type = 7)
  summary_rows[[name]] <- list(
    col_summary = col_summary,
    meta = data.table(
      config = name, N = prep$N, min_period = prep$min_period, max_period = prep$max_period,
      n_ids = uniqueN(d$id), n_trials = uniqueN(d$trial_period),
      w_q000 = q[[1]], w_q001 = q[[2]], w_q025 = q[[3]], w_q050 = q[[4]],
      w_q075 = q[[5]], w_q099 = q[[6]], w_q100 = q[[7]], w_mean = mean(d$weight, na.rm = TRUE)
    )
  )

  # Per trial-period checksums.
  by_trial <- d[, c(
    list(n = .N, n_ids = uniqueN(id), sum_weight = sum(weight), sum_weight_sq = sum(weight^2),
         sum_outcome = sum(outcome, na.rm = TRUE), sum_followup = sum(followup_time),
         sum_treatment = sum(treatment)),
    if ("assigned_treatment" %in% names(d)) list(sum_assigned = sum(assigned_treatment)) else list(),
    if ("dose" %in% names(d)) list(sum_dose = sum(dose)) else list()
  ), by = trial_period]
  setorder(by_trial, trial_period)
  fwrite(by_trial, file.path(out_dir, paste0(name, "_by_trial.csv")))

  # Deterministic row sample in (id, trial_period, followup_time) order.
  idx <- unique(round(seq(1, n, length.out = min(n, 300))))
  fwrite(d[idx], file.path(out_dir, paste0(name, "_sample_rows.csv")))

  # Weight-model coefficient tables.
  wm <- c(prep$switch_models, prep$censor_models)
  if (length(wm)) {
    wm_tab <- rbindlist(lapply(names(wm), function(m) {
      s <- as.data.table(wm[[m]]$summary)
      s[, model := m]
      s[, nobs := wm[[m]]$fit_summary$nobs]
      s
    }))
    fwrite(wm_tab, file.path(out_dir, paste0(name, "_weight_models.csv")))
  }

  if (is.null(cfg$msm)) next
  # Marginal structural model.
  msm <- do.call(
    trial_msm,
    c(list(data = prep, use_sample_weights = FALSE, quiet = TRUE), cfg$msm)
  )
  rs <- as.data.table(msm$robust$summary)
  # trial_msm fits on prep$data in its original row order; the expanded data
  # has no missing model variables, so the cluster vector aligns row for row.
  stopifnot(nobs(msm$model) == nrow(prep$data))
  vc_hc0 <- sandwich::vcovCL(msm$model, cluster = prep$data$id, type = "HC0", cadjust = FALSE)
  rs[, robust_se_no_cadjust := sqrt(diag(vc_hc0))[names]]
  rs[, model_se := sqrt(diag(vcov(msm$model)))[names]]
  rs[, n_obs := nobs(msm$model)]
  rs[, n_clusters := uniqueN(d$id)]
  rs[, formula := paste(deparse(msm$args$model_formula), collapse = " ")]
  rs[, config := name]
  fwrite(rs, file.path(out_dir, paste0(name, "_msm.csv")))
}

fwrite(rbindlist(lapply(summary_rows, `[[`, "col_summary")), file.path(out_dir, "expanded_column_summary.csv"))
fwrite(rbindlist(lapply(summary_rows, `[[`, "meta")), file.path(out_dir, "expanded_meta.csv"))
fwrite(as.data.table(data_censored), file.path(out_dir, "data_censored.csv"))
writeLines(
  c(
    paste("TrialEmulation", as.character(packageVersion("TrialEmulation"))),
    "git f432103fe20dc65aac6e3efaca797d1b98ff364f",
    paste("data.table", as.character(packageVersion("data.table"))),
    paste("sandwich", as.character(packageVersion("sandwich"))),
    R.version.string
  ),
  file.path(out_dir, "VERSIONS.txt")
)
message("done")
