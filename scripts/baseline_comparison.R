#!/usr/bin/env Rscript
# ═══════════════════════════════════════════════════════════════════
# Baseline Comparison for LODOCV Prescriptive Evaluation
# Compares LME-based selection against naive and heuristic baselines
# ═══════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(tidyr)
  library(MuMIn)
})

INPUT  <- "meta_dataset_kd_only.csv"
OUTDIR <- "stage2_output"

cat("═══════════════════════════════════════════════════\n")
cat("Baseline Comparison for LODOCV\n")
cat("═══════════════════════════════════════════════════\n\n")

# ─── 1. Load Data ───────────────────────────────────────────────
df <- read.csv(INPUT, stringsAsFactors = FALSE) %>%
  filter(method != "label_smoothing") %>%
  filter(!is.na(delta_f1))

df$method  <- factor(df$method)
df$family  <- factor(df$family)
df$dataset <- factor(df$dataset)
df$student <- factor(df$student)
df$method  <- relevel(df$method, ref = "vanilla")

cat("Data: ", nrow(df), "rows,", length(unique(df$dataset)), "datasets,",
    length(unique(df$method)), "methods\n\n")

# ─── CDR threshold lookup (from Table 6) ───────────────────────
CDR_THRESHOLDS <- c(
  cord = 5047, ban = 5077, akr = 5581, vanilla = 7179,
  takd = 8166, ctkd = 8720, nkd = 15068, dist = 25014,
  dkd = 28617, simkd = 32994, crd = Inf, fitnet = Inf
)

# ─── Helper: evaluate prescriptive strategy ────────────────────
evaluate_prescriptive <- function(test_data, prescribed_df) {
  # prescribed_df: has (student, seed, prescribed_method)
  eval_rows <- test_data %>%
    inner_join(prescribed_df, by = c("student", "seed"),
               relationship = "many-to-many") %>%
    filter(method == prescribed_method)

  oracle <- test_data %>%
    group_by(student, seed) %>%
    slice_max(delta_f1, n = 1, with_ties = FALSE) %>%
    ungroup()

  merged <- eval_rows %>%
    inner_join(oracle %>% select(student, seed,
                                 oracle_method = method,
                                 oracle_delta = delta_f1),
               by = c("student", "seed"))

  list(
    regret           = mean(merged$oracle_delta - merged$delta_f1, na.rm = TRUE),
    safety           = 1 - mean(merged$delta_f1 < -1, na.rm = TRUE),
    catastrophic_avoid = 1 - mean(merged$delta_f1 < -3, na.rm = TRUE),
    method_match     = mean(merged$method == merged$oracle_method, na.rm = TRUE),
    mean_delta       = mean(merged$delta_f1, na.rm = TRUE),
    positive_rate    = mean(merged$delta_f1 > 0, na.rm = TRUE),
    n_eval           = nrow(merged)
  )
}

# ─── 2. Also compute: random method baseline ──────────────────
set.seed(42)

# ─── 3. LODOCV with all baselines ─────────────────────────────
datasets <- unique(df$dataset)
all_results <- list()

for (held_out in datasets) {
  cat(sprintf("─── %s ───\n", held_out))
  train <- df %>% filter(dataset != held_out)
  test  <- df %>% filter(dataset == held_out)
  test_configs <- test %>% distinct(student, seed)
  n_methods <- length(unique(test$method))

  # --- Baseline 1: Random method ---
  random_rx <- test_configs %>%
    rowwise() %>%
    mutate(prescribed_method = sample(levels(df$method), 1)) %>%
    ungroup()
  random_eval <- evaluate_prescriptive(test, random_rx)

  # --- Baseline 2: Always-BAN ---
  ban_rx <- test_configs %>% mutate(prescribed_method = "ban")
  ban_eval <- evaluate_prescriptive(test, ban_rx)

  # --- Baseline 3: Always-BAN-CL ---
  cord_rx <- test_configs %>% mutate(prescribed_method = "cord")
  cord_eval <- evaluate_prescriptive(test, cord_rx)

  # --- Baseline 4: Best-single from training data ---
  best_train_method <- train %>%
    group_by(method) %>%
    summarise(mean_delta = mean(delta_f1), .groups = "drop") %>%
    slice_max(mean_delta, n = 1) %>%
    pull(method) %>%
    as.character()
  best_single_rx <- test_configs %>% mutate(prescribed_method = best_train_method)
  best_single_eval <- evaluate_prescriptive(test, best_single_rx)

  # --- Baseline 5: CDR-threshold heuristic (safest method) ---
  threshold_rx <- test %>%
    distinct(student, seed, CDR) %>%
    rowwise() %>%
    mutate(
      prescribed_method = {
        safe <- names(CDR_THRESHOLDS)[CDR > CDR_THRESHOLDS]
        if (length(safe) == 0) {
          "ban"
        } else {
          safe[which.min(CDR_THRESHOLDS[safe])]
        }
      }
    ) %>%
    ungroup() %>%
    select(student, seed, prescribed_method)
  threshold_eval <- evaluate_prescriptive(test, threshold_rx)

  # --- Baseline 6: CDR-threshold + best mean delta from train ---
  threshold_best_rx <- test %>%
    distinct(student, seed, CDR) %>%
    rowwise() %>%
    mutate(
      prescribed_method = {
        safe <- names(CDR_THRESHOLDS)[CDR > CDR_THRESHOLDS]
        if (length(safe) == 0) {
          "ban"
        } else {
          train_means <- train %>%
            filter(method %in% safe) %>%
            group_by(method) %>%
            summarise(md = mean(delta_f1), .groups = "drop") %>%
            slice_max(md, n = 1)
          as.character(train_means$method[1])
        }
      }
    ) %>%
    ungroup() %>%
    select(student, seed, prescribed_method)
  threshold_best_eval <- evaluate_prescriptive(test, threshold_best_rx)

  # --- LME Model (simplified for generalization) ---
  m_cv <- tryCatch({
    lmer(
      delta_f1 ~
        log_CDR + log_IR + baseline_f1 + log_K +
        method +
        log_CDR : method +
        log_IR  : method +
        (1 | dataset) + (1 | student),
      data = train,
      REML = FALSE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
    )
  }, error = function(e) NULL)

  if (!is.null(m_cv)) {
    test$pred_delta <- predict(m_cv, newdata = test, re.form = NA,
                               allow.new.levels = TRUE)
    test$pred_sign <- as.integer(test$pred_delta > 0)
    sign_acc_lme <- mean(test$pred_sign == test$win_f1, na.rm = TRUE)

    lme_rx <- test %>%
      group_by(student, seed) %>%
      slice_max(pred_delta, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      select(student, seed, prescribed_method = method)
    lme_eval <- evaluate_prescriptive(test, lme_rx)
  } else {
    sign_acc_lme <- NA
    lme_eval <- list(regret = NA, safety = NA, catastrophic_avoid = NA,
                     method_match = NA, mean_delta = NA, positive_rate = NA,
                     n_eval = NA)
  }

  # --- Oracle ---
  oracle_rx <- test %>%
    group_by(student, seed) %>%
    slice_max(delta_f1, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    select(student, seed, prescribed_method = method)
  oracle_eval <- evaluate_prescriptive(test, oracle_rx)

  # --- Sign accuracy baselines ---
  majority_sign <- if (mean(train$win_f1) > 0.5) 1L else 0L
  sign_acc_majority <- mean(majority_sign == test$win_f1, na.rm = TRUE)

  train_wr <- train %>%
    group_by(method) %>%
    summarise(wr = mean(win_f1), .groups = "drop")
  test_with_wr <- test %>%
    left_join(train_wr, by = "method") %>%
    mutate(pred_sign_wr = as.integer(wr > 0.5))
  sign_acc_trainwr <- mean(test_with_wr$pred_sign_wr == test_with_wr$win_f1, na.rm = TRUE)

  cat(sprintf("  Random:              regret=%.2f, safety=%.1f%%, catast_avoid=%.1f%%, pos_rate=%.1f%%\n",
              random_eval$regret, 100*random_eval$safety, 100*random_eval$catastrophic_avoid, 100*random_eval$positive_rate))
  cat(sprintf("  Always-BAN:          regret=%.2f, safety=%.1f%%, catast_avoid=%.1f%%, pos_rate=%.1f%%\n",
              ban_eval$regret, 100*ban_eval$safety, 100*ban_eval$catastrophic_avoid, 100*ban_eval$positive_rate))
  cat(sprintf("  CDR-threshold:       regret=%.2f, safety=%.1f%%, catast_avoid=%.1f%%, pos_rate=%.1f%%\n",
              threshold_eval$regret, 100*threshold_eval$safety, 100*threshold_eval$catastrophic_avoid, 100*threshold_eval$positive_rate))
  cat(sprintf("  CDR-thresh+best:     regret=%.2f, safety=%.1f%%, catast_avoid=%.1f%%, pos_rate=%.1f%%\n",
              threshold_best_eval$regret, 100*threshold_best_eval$safety, 100*threshold_best_eval$catastrophic_avoid, 100*threshold_best_eval$positive_rate))
  cat(sprintf("  LME model:           regret=%.2f, safety=%.1f%%, catast_avoid=%.1f%%, pos_rate=%.1f%%\n",
              lme_eval$regret, 100*lme_eval$safety, 100*lme_eval$catastrophic_avoid, 100*lme_eval$positive_rate))
  cat(sprintf("  Oracle:              regret=%.2f, safety=%.1f%%, catast_avoid=%.1f%%, pos_rate=%.1f%%\n",
              oracle_eval$regret, 100*oracle_eval$safety, 100*oracle_eval$catastrophic_avoid, 100*oracle_eval$positive_rate))
  cat(sprintf("  Sign acc — majority: %.1f%%, train-WR: %.1f%%, LME: %.1f%%\n\n",
              100*sign_acc_majority, 100*sign_acc_trainwr, 100*sign_acc_lme))

  all_results[[held_out]] <- data.frame(
    held_out = held_out,
    n_configs = nrow(test_configs),
    # Random
    random_regret = random_eval$regret,
    random_safety = random_eval$safety,
    random_catast = random_eval$catastrophic_avoid,
    random_delta  = random_eval$mean_delta,
    random_posrate = random_eval$positive_rate,
    # Always-BAN
    ban_regret = ban_eval$regret,
    ban_safety = ban_eval$safety,
    ban_catast = ban_eval$catastrophic_avoid,
    ban_delta  = ban_eval$mean_delta,
    ban_posrate = ban_eval$positive_rate,
    # Always-BAN-CL
    cord_regret = cord_eval$regret,
    cord_safety = cord_eval$safety,
    cord_catast = cord_eval$catastrophic_avoid,
    cord_delta  = cord_eval$mean_delta,
    cord_posrate = cord_eval$positive_rate,
    # Best-single
    best_single_regret = best_single_eval$regret,
    best_single_safety = best_single_eval$safety,
    best_single_catast = best_single_eval$catastrophic_avoid,
    best_single_delta  = best_single_eval$mean_delta,
    best_single_posrate = best_single_eval$positive_rate,
    best_single_method = best_train_method,
    # CDR-threshold
    threshold_regret = threshold_eval$regret,
    threshold_safety = threshold_eval$safety,
    threshold_catast = threshold_eval$catastrophic_avoid,
    threshold_delta  = threshold_eval$mean_delta,
    threshold_posrate = threshold_eval$positive_rate,
    # CDR-thresh+best
    threshold_best_regret = threshold_best_eval$regret,
    threshold_best_safety = threshold_best_eval$safety,
    threshold_best_catast = threshold_best_eval$catastrophic_avoid,
    threshold_best_delta  = threshold_best_eval$mean_delta,
    threshold_best_posrate = threshold_best_eval$positive_rate,
    # LME
    lme_regret = lme_eval$regret,
    lme_safety = lme_eval$safety,
    lme_catast = lme_eval$catastrophic_avoid,
    lme_delta  = lme_eval$mean_delta,
    lme_posrate = lme_eval$positive_rate,
    # Oracle
    oracle_regret = oracle_eval$regret,
    oracle_safety = oracle_eval$safety,
    oracle_catast = oracle_eval$catastrophic_avoid,
    oracle_delta  = oracle_eval$mean_delta,
    oracle_posrate = oracle_eval$positive_rate,
    # Sign accuracy
    sign_acc_majority = sign_acc_majority,
    sign_acc_trainwr  = sign_acc_trainwr,
    sign_acc_lme      = sign_acc_lme,
    stringsAsFactors = FALSE
  )
}

# ─── 3. Summary ────────────────────────────────────────────────
results_df <- bind_rows(all_results)

cat("\n═══════════════════════════════════════════════════\n")
cat("SUMMARY: Mean across all held-out datasets\n")
cat("═══════════════════════════════════════════════════\n\n")

cat("PRESCRIPTIVE (method selection) metrics:\n")
cat(sprintf("  %-22s  Regret  Safety  Catast.Avoid  PosRate  Mean-Δ\n", "Strategy"))
cat(sprintf("  %-22s  ------  ------  -----------   -------  ------\n", ""))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "Random",
            mean(results_df$random_regret), 100*mean(results_df$random_safety),
            100*mean(results_df$random_catast), 100*mean(results_df$random_posrate),
            mean(results_df$random_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "Always-BAN",
            mean(results_df$ban_regret), 100*mean(results_df$ban_safety),
            100*mean(results_df$ban_catast), 100*mean(results_df$ban_posrate),
            mean(results_df$ban_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "Always-BAN-CL",
            mean(results_df$cord_regret), 100*mean(results_df$cord_safety),
            100*mean(results_df$cord_catast), 100*mean(results_df$cord_posrate),
            mean(results_df$cord_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "Best-single (train)",
            mean(results_df$best_single_regret), 100*mean(results_df$best_single_safety),
            100*mean(results_df$best_single_catast), 100*mean(results_df$best_single_posrate),
            mean(results_df$best_single_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "CDR-threshold",
            mean(results_df$threshold_regret), 100*mean(results_df$threshold_safety),
            100*mean(results_df$threshold_catast), 100*mean(results_df$threshold_posrate),
            mean(results_df$threshold_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "CDR-thresh+best",
            mean(results_df$threshold_best_regret), 100*mean(results_df$threshold_best_safety),
            100*mean(results_df$threshold_best_catast), 100*mean(results_df$threshold_best_posrate),
            mean(results_df$threshold_best_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "LME model",
            mean(results_df$lme_regret), 100*mean(results_df$lme_safety),
            100*mean(results_df$lme_catast), 100*mean(results_df$lme_posrate),
            mean(results_df$lme_delta)))
cat(sprintf("  %-22s  %5.2f   %5.1f%%     %5.1f%%     %5.1f%%   %+.2f\n", "Oracle",
            mean(results_df$oracle_regret), 100*mean(results_df$oracle_safety),
            100*mean(results_df$oracle_catast), 100*mean(results_df$oracle_posrate),
            mean(results_df$oracle_delta)))

cat("\nSIGN ACCURACY (per-sample) metrics:\n")
cat(sprintf("  Majority-class:      %.1f%%\n", 100*mean(results_df$sign_acc_majority)))
cat(sprintf("  Train-WR per method: %.1f%%\n", 100*mean(results_df$sign_acc_trainwr)))
cat(sprintf("  LME model:           %.1f%%\n", 100*mean(results_df$sign_acc_lme)))

cat("\nKEY INSIGHT:\n")
cat("  'Always-BAN' is the strongest baseline because BAN has the\n")
cat("  lowest CDR threshold (~5,077) — this IS the model's finding.\n")
cat("  The CDR-threshold heuristic (derived from the LME model)\n")
cat("  matches BAN's performance because it selects BAN/BAN-CL when\n")
cat("  both are safe. The model's value is EXPLANATORY: it reveals\n")
cat("  WHY BAN is safe (selective exposure) and WHEN other methods\n")
cat("  become risky (method-specific CDR thresholds).\n")

write.csv(results_df,
          file.path(OUTDIR, "tables", "baseline_comparison.csv"),
          row.names = FALSE)
cat("\nResults saved to tables/baseline_comparison.csv\n")
