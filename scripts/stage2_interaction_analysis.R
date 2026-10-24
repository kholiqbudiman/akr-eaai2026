#!/usr/bin/env Rscript
# ═══════════════════════════════════════════════════════════════════
# Stage 2: Interaction Modeling for CDR-Guided Method Selection
# Input:  meta_dataset_kd_only.csv (6,648 runs, 13 KD methods)
# Output: tables/ and figures/ in the output directory
#
# Usage:
#   Rscript stage2_interaction_analysis.R
#   Rscript stage2_interaction_analysis.R --input /path/to/meta_dataset_kd_only.csv
# ═══════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(rlang)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(broom.mixed)
  library(MuMIn)
})

# ─── Config ──────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
INPUT  <- if (length(args) >= 2 && args[1] == "--input") args[2] else
          "meta_dataset_kd_only.csv"
OUTDIR <- "stage2_output"

dir.create(file.path(OUTDIR, "tables"),  recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTDIR, "figures"), recursive = TRUE, showWarnings = FALSE)

cat("═══════════════════════════════════════════════════\n")
cat("Stage 2: Interaction Modeling\n")
cat("═══════════════════════════════════════════════════\n")
cat("Input:", INPUT, "\n")
cat("Output:", OUTDIR, "\n\n")

# ─── 1. Load & Prepare ──────────────────────────────────────────
cat("1. Loading data...\n")
df <- read.csv(INPUT, stringsAsFactors = FALSE)
cat("   Rows:", nrow(df), " Methods:", length(unique(df$method)),
    " Datasets:", length(unique(df$dataset)), "\n")

# Remove label_smoothing (regularizer, not KD)
df <- df %>% filter(method != "label_smoothing")
cat("   After removing label_smoothing:", nrow(df), "rows\n")

# Factor encoding
df$method  <- factor(df$method)
df$family  <- factor(df$family)
df$dataset <- factor(df$dataset)
df$student <- factor(df$student)

# Relevel: vanilla as reference (most common baseline KD)
df$method <- relevel(df$method, ref = "vanilla")

# Remove rows with missing delta_f1
df <- df %>% filter(!is.na(delta_f1))
cat("   Complete cases:", nrow(df), "\n\n")

# Quick descriptive table
cat("Per-method summary:\n")
method_summary <- df %>%
  group_by(method, family) %>%
  summarise(
    N      = n(),
    mean_d = mean(delta_f1),
    sd_d   = sd(delta_f1),
    WR     = 100 * mean(win_f1),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_d))
print(as.data.frame(method_summary), row.names = FALSE)
cat("\n")


# ═══════════════════════════════════════════════════════════════════
# 2. Model 1: Linear Mixed-Effects (interpretable)
# ═══════════════════════════════════════════════════════════════════
cat("2. Fitting linear mixed-effects model...\n")

m1 <- lmer(
  delta_f1 ~
    log_CDR + log_IR + baseline_f1 + log_K +
    method +
    log_CDR : method +
    log_IR  : method +
    log_CDR : log_IR +
    baseline_f1 : method +
    (1 | dataset) + (1 | student),
  data = df,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
)

cat("   Converged:", is.null(m1@optinfo$conv$lme4), "\n")
cat("   Marginal R²:    ", round(r.squaredGLMM(m1)[1, "R2m"], 4), "\n")
cat("   Conditional R²: ", round(r.squaredGLMM(m1)[1, "R2c"], 4), "\n\n")

# Save full model summary
sink(file.path(OUTDIR, "tables", "model1_summary.txt"))
cat("Linear Mixed-Effects Model Summary\n")
cat("══════════════════════════════════\n\n")
print(summary(m1))
cat("\n\nR-squared (Nakagawa):\n")
print(r.squaredGLMM(m1))
sink()
cat("   Model summary saved to tables/model1_summary.txt\n")


# ═══════════════════════════════════════════════════════════════════
# 3. ANOVA of Interaction Terms
# ═══════════════════════════════════════════════════════════════════
cat("\n3. ANOVA (Type III) of interaction terms...\n")

anova_res <- anova(m1, type = 3, ddf = "Satterthwaite")
cat("\n")
print(anova_res)

# Save ANOVA table
sink(file.path(OUTDIR, "tables", "anova_interactions.txt"))
cat("Type III ANOVA — Interaction Terms\n")
cat("═══════════════════════════════════\n\n")
print(anova_res)
sink()

# Also save as CSV
anova_df <- as.data.frame(anova_res)
anova_df$Term <- rownames(anova_df)
write.csv(anova_df, file.path(OUTDIR, "tables", "anova_interactions.csv"),
          row.names = FALSE)
cat("   ANOVA tables saved.\n")


# ═══════════════════════════════════════════════════════════════════
# 4. Per-Method CDR Slopes (emtrends)
# ═══════════════════════════════════════════════════════════════════
cat("\n4. Extracting per-method CDR slopes...\n")

cdr_trends <- emtrends(m1, ~ method, var = "log_CDR")
cdr_trends_df <- as.data.frame(cdr_trends)
cdr_trends_df <- cdr_trends_df %>% arrange(desc(log_CDR.trend))

cat("\nPer-method CDR slopes (how much delta_f1 changes per log10 unit of CDR):\n")
print(cdr_trends_df, row.names = FALSE)

write.csv(cdr_trends_df,
          file.path(OUTDIR, "tables", "cdr_slopes_per_method.csv"),
          row.names = FALSE)


# ═══════════════════════════════════════════════════════════════════
# 5. Per-Method IR Slopes (emtrends)
# ═══════════════════════════════════════════════════════════════════
cat("\n5. Extracting per-method IR slopes...\n")

ir_trends <- emtrends(m1, ~ method, var = "log_IR")
ir_trends_df <- as.data.frame(ir_trends)
ir_trends_df <- ir_trends_df %>% arrange(desc(log_IR.trend))

cat("\nPer-method IR slopes (how much delta_f1 changes per log10 unit of IR):\n")
print(ir_trends_df, row.names = FALSE)

write.csv(ir_trends_df,
          file.path(OUTDIR, "tables", "ir_slopes_per_method.csv"),
          row.names = FALSE)


# ═══════════════════════════════════════════════════════════════════
# 6. CDR Threshold per Method (crossing point)
# ═══════════════════════════════════════════════════════════════════
cat("\n6. Computing CDR threshold (zero-crossing) per method...\n")

compute_crossing <- function(model, method_name, ir_val = 0.5, baseline_val = NULL) {
  # Predict delta_f1 across CDR range for given method
  if (is.null(baseline_val)) baseline_val <- median(df$baseline_f1)
  newdata <- data.frame(
    log_CDR    = seq(1.5, 5.0, by = 0.01),
    log_IR     = ir_val,
    baseline_f1 = baseline_val,
    log_K      = median(df$log_K),
    method     = factor(method_name, levels = levels(df$method)),
    dataset    = NA,
    student    = NA
  )
  newdata$pred <- predict(m1, newdata, re.form = NA)

  # Find zero crossing
  sign_changes <- which(diff(sign(newdata$pred)) != 0)
  if (length(sign_changes) > 0) {
    idx <- sign_changes[1]
    # Linear interpolation
    x1 <- newdata$log_CDR[idx]; x2 <- newdata$log_CDR[idx + 1]
    y1 <- newdata$pred[idx];    y2 <- newdata$pred[idx + 1]
    crossing_log <- x1 - y1 * (x2 - x1) / (y2 - y1)
    return(10^crossing_log)
  }
  # Check if always positive or always negative
  if (all(newdata$pred > 0)) return(0)   # always positive
  if (all(newdata$pred < 0)) return(Inf) # never positive
  return(NA)
}

METHOD_FAMILY <- c(
  vanilla = "logit", takd = "logit", dkd = "logit",
  nkd = "logit", dist = "logit", ctkd = "logit",
  fitnet = "feature", simkd = "feature", crd = "relation",
  akr = "adaptive", ban = "self_distill", cord = "self_distill"
)

methods_list <- levels(df$method)
crossing_df <- data.frame(
  method = methods_list,
  CDR_threshold = sapply(methods_list, function(m) compute_crossing(m1, m)),
  stringsAsFactors = FALSE
)
crossing_df$family <- METHOD_FAMILY[crossing_df$method]
crossing_df <- crossing_df %>% arrange(CDR_threshold)

cat("\nCDR threshold per method (CDR where delta_f1 crosses zero):\n")
cat("  (CDR* = 0 means always positive; Inf means never positive)\n\n")
print(crossing_df, row.names = FALSE)

write.csv(crossing_df,
          file.path(OUTDIR, "tables", "cdr_threshold_per_method.csv"),
          row.names = FALSE)


# ═══════════════════════════════════════════════════════════════════
# 7. Figure: CDR × Method Interaction
# ═══════════════════════════════════════════════════════════════════
cat("\n7. Generating figures...\n")

# 7a. CDR × Method predicted delta_f1
key_methods <- c("vanilla", "dkd", "akr", "ban", "fitnet", "takd")
key_methods <- key_methods[key_methods %in% levels(df$method)]

pred_cdr <- expand.grid(
  log_CDR     = seq(1.5, 5.0, by = 0.05),
  log_IR      = median(df$log_IR),
  baseline_f1 = median(df$baseline_f1),
  log_K       = median(df$log_K),
  method      = factor(key_methods, levels = levels(df$method))
)
pred_cdr$dataset <- NA
pred_cdr$student <- NA
pred_cdr$pred    <- predict(m1, pred_cdr, re.form = NA)
pred_cdr$CDR     <- 10^pred_cdr$log_CDR

p_cdr <- ggplot(pred_cdr, aes(x = CDR, y = pred, color = method)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_x_log10(
    breaks = c(100, 1000, 10000, 100000),
    labels = c("100", "1K", "10K", "100K")
  ) +
  labs(
    x = "CDR (log scale)",
    y = expression(Predicted ~ Delta * F1 ~ "(pp)"),
    color = "Method",
    title = "CDR-dependent KD effectiveness by method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = c(0.85, 0.25),
    legend.background = element_rect(fill = "white", color = "gray80"),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(OUTDIR, "figures", "fig_cdr_method_interaction.pdf"),
       p_cdr, width = 7, height = 4.5)
cat("   fig_cdr_method_interaction.pdf saved.\n")


# 7b. CDR × IR Interaction (4-panel by method)
pred_cdr_ir <- expand.grid(
  log_CDR     = seq(1.5, 5.0, by = 0.1),
  log_IR      = c(0, 0.3, 0.7, 1.0, 1.5, 1.78),
  baseline_f1 = median(df$baseline_f1),
  log_K       = median(df$log_K),
  method      = factor(c("vanilla", "dkd", "akr", "ban"),
                       levels = levels(df$method))
)
pred_cdr_ir$dataset <- NA
pred_cdr_ir$student <- NA
pred_cdr_ir$pred    <- predict(m1, pred_cdr_ir, re.form = NA)
pred_cdr_ir$CDR     <- 10^pred_cdr_ir$log_CDR
pred_cdr_ir$IR      <- round(10^pred_cdr_ir$log_IR, 1)
pred_cdr_ir$IR_label <- paste0("IR=", pred_cdr_ir$IR)

p_cdr_ir <- ggplot(pred_cdr_ir, aes(x = CDR, y = pred,
                                     color = factor(IR))) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ method, scales = "free_y", ncol = 2) +
  scale_x_log10(
    breaks = c(100, 1000, 10000, 100000),
    labels = c("100", "1K", "10K", "100K")
  ) +
  scale_color_viridis_d(option = "plasma", end = 0.85) +
  labs(
    x = "CDR (log scale)",
    y = expression(Predicted ~ Delta * F1 ~ "(pp)"),
    color = "IR",
    title = "Joint CDR x IR effect on KD transfer"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(OUTDIR, "figures", "fig_cdr_ir_interaction.pdf"),
       p_cdr_ir, width = 8, height = 6)
cat("   fig_cdr_ir_interaction.pdf saved.\n")


# 7c. CDR Slopes Forest Plot
slopes_combined <- bind_rows(
  cdr_trends_df %>%
    rename(trend = log_CDR.trend, SE_val = SE, lower = asymp.LCL, upper = asymp.UCL) %>%
    mutate(factor_type = "CDR slope"),
  ir_trends_df %>%
    rename(trend = log_IR.trend, SE_val = SE, lower = asymp.LCL, upper = asymp.UCL) %>%
    mutate(factor_type = "IR slope")
)

p_slopes <- ggplot(slopes_combined,
                   aes(x = trend, y = reorder(method, trend),
                       xmin = lower, xmax = upper)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  geom_errorbarh(height = 0.3, linewidth = 0.5) +
  geom_point(size = 2.5) +
  facet_wrap(~ factor_type, scales = "free_x") +
  labs(
    x = "Slope (pp per log10 unit)",
    y = NULL,
    title = "Per-method sensitivity to CDR and IR"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(OUTDIR, "figures", "fig_slopes_forest.pdf"),
       p_slopes, width = 8, height = 5)
cat("   fig_slopes_forest.pdf saved.\n")


# 7d. CDR Threshold Bar Chart
crossing_plot <- crossing_df %>%
  filter(is.finite(CDR_threshold) & CDR_threshold > 0) %>%
  mutate(method = reorder(method, CDR_threshold))

p_threshold <- ggplot(crossing_plot,
                      aes(x = CDR_threshold, y = method, fill = family)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.0f", CDR_threshold)),
            hjust = -0.1, size = 3.2) +
  scale_x_log10(
    breaks = c(10, 100, 1000, 10000, 100000),
    labels = c("10", "100", "1K", "10K", "100K"),
    expand = expansion(mult = c(0, 0.3))
  ) +
  labs(
    x = "CDR threshold (where delta F1 crosses zero)",
    y = NULL,
    fill = "Family",
    title = "Minimum CDR for positive transfer by method"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(OUTDIR, "figures", "fig_cdr_threshold.pdf"),
       p_threshold, width = 7, height = 4)
cat("   fig_cdr_threshold.pdf saved.\n")


# ═══════════════════════════════════════════════════════════════════
# 8. Model 2: Binary Classification (win/lose prediction)
# ═══════════════════════════════════════════════════════════════════
cat("\n8. Fitting binary win/lose model (GLMM)...\n")

m2 <- glmer(
  win_f1 ~
    log_CDR + log_IR + baseline_f1 + log_K +
    method +
    log_CDR : method +
    log_IR  : method +
    (1 | dataset) + (1 | student),
  data = df,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

cat("   Converged:", is.null(m2@optinfo$conv$lme4), "\n")

# In-sample classification accuracy
df$pred_win <- as.integer(predict(m2, type = "response") > 0.5)
sign_acc <- mean(df$pred_win == df$win_f1, na.rm = TRUE)
cat("   In-sample sign accuracy:", round(100 * sign_acc, 1), "%\n")

sink(file.path(OUTDIR, "tables", "model2_glmm_summary.txt"))
cat("Binary GLMM: Win/Lose Prediction\n")
cat("═════════════════════════════════\n\n")
print(summary(m2))
cat("\nIn-sample sign accuracy:", round(100 * sign_acc, 1), "%\n")
sink()
cat("   GLMM summary saved.\n")


# ═══════════════════════════════════════════════════════════════════
# 9. Leave-One-Dataset-Out Cross-Validation
# ═══════════════════════════════════════════════════════════════════
cat("\n9. Leave-one-dataset-out cross-validation...\n")

datasets <- unique(df$dataset)
lodocv_results <- list()

for (held_out in datasets) {
  train <- df %>% filter(dataset != held_out)
  test  <- df %>% filter(dataset == held_out)

  # Fit on training data
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

  if (is.null(m_cv)) {
    cat("   ", held_out, ": model failed to converge, skipping\n")
    next
  }

  # Predict on held-out dataset
  test$pred_delta <- predict(m_cv, newdata = test, re.form = NA,
                             allow.new.levels = TRUE)

  # Sign accuracy
  test$pred_sign <- as.integer(test$pred_delta > 0)
  sign_acc_cv <- mean(test$pred_sign == test$win_f1, na.rm = TRUE)

  # Prescriptive evaluation: for each (student, seed), which method does model pick?
  prescribed <- test %>%
    group_by(student, seed) %>%
    slice_max(pred_delta, n = 1, with_ties = FALSE) %>%
    ungroup()

  oracle <- test %>%
    group_by(student, seed) %>%
    slice_max(delta_f1, n = 1, with_ties = FALSE) %>%
    ungroup()

  mean_regret <- mean(oracle$delta_f1 - prescribed$delta_f1, na.rm = TRUE)
  method_match <- mean(prescribed$method == oracle$method, na.rm = TRUE)

  # Safety: how many prescribed methods are significantly harmful?
  prescribed_harmful <- mean(prescribed$delta_f1 < -1, na.rm = TRUE)

  lodocv_results[[held_out]] <- data.frame(
    held_out       = held_out,
    n_test         = nrow(test),
    sign_accuracy  = round(sign_acc_cv, 4),
    mean_regret_pp = round(mean_regret, 4),
    method_match   = round(method_match, 4),
    safety_rate    = round(1 - prescribed_harmful, 4)
  )
  cat(sprintf("   %-15s: sign_acc=%.1f%%, regret=%.2f pp, safe=%.1f%%\n",
              held_out, 100*sign_acc_cv, mean_regret, 100*(1-prescribed_harmful)))
}

lodocv_df <- bind_rows(lodocv_results)

cat("\nLODOCV Summary:\n")
cat(sprintf("  Mean sign accuracy: %.1f%%\n", 100 * mean(lodocv_df$sign_accuracy)))
cat(sprintf("  Mean regret:        %.2f pp\n", mean(lodocv_df$mean_regret_pp)))
cat(sprintf("  Mean safety rate:   %.1f%%\n", 100 * mean(lodocv_df$safety_rate)))

write.csv(lodocv_df, file.path(OUTDIR, "tables", "lodocv_results.csv"),
          row.names = FALSE)
cat("   LODOCV results saved.\n")


# ═══════════════════════════════════════════════════════════════════
# 10. Pairwise Method Contrasts
# ═══════════════════════════════════════════════════════════════════
cat("\n10. Pairwise method contrasts (at median CDR/IR)...\n")

em_methods <- emmeans(m1, ~ method,
                      at = list(log_CDR = median(df$log_CDR),
                                log_IR  = median(df$log_IR),
                                baseline_f1 = median(df$baseline_f1),
                                log_K = median(df$log_K)))
contrasts <- pairs(em_methods, adjust = "BH")

sink(file.path(OUTDIR, "tables", "pairwise_contrasts.txt"))
cat("Pairwise Method Contrasts (Benjamini-Hochberg adjusted)\n")
cat("At median CDR, IR, baseline_f1\n")
cat("══════════════════════════════════════════════════════\n\n")
print(summary(contrasts))
sink()

contrasts_df <- as.data.frame(summary(contrasts))
write.csv(contrasts_df, file.path(OUTDIR, "tables", "pairwise_contrasts.csv"),
          row.names = FALSE)
cat("   Pairwise contrasts saved.\n")


# ═══════════════════════════════════════════════════════════════════
# 11. Summary Report
# ═══════════════════════════════════════════════════════════════════
cat("\n")
cat("═══════════════════════════════════════════════════\n")
cat("ANALYSIS COMPLETE\n")
cat("═══════════════════════════════════════════════════\n")
cat("\nOutput files:\n")
cat("  tables/\n")
cat("    model1_summary.txt         — LME model coefficients\n")
cat("    model2_glmm_summary.txt    — Binary GLMM summary\n")
cat("    anova_interactions.csv     — Type III ANOVA\n")
cat("    cdr_slopes_per_method.csv  — CDR sensitivity per method\n")
cat("    ir_slopes_per_method.csv   — IR sensitivity per method\n")
cat("    cdr_threshold_per_method.csv — Zero-crossing CDR*\n")
cat("    pairwise_contrasts.csv     — BH-corrected contrasts\n")
cat("    lodocv_results.csv         — Cross-validation metrics\n")
cat("  figures/\n")
cat("    fig_cdr_method_interaction.pdf  — CDR x Method lines\n")
cat("    fig_cdr_ir_interaction.pdf      — CDR x IR x Method panels\n")
cat("    fig_slopes_forest.pdf           — CDR/IR slope forest plot\n")
cat("    fig_cdr_threshold.pdf           — CDR threshold bar chart\n")
cat("\n")
cat("Key results for paper:\n")
cat("  1. Table: CDR threshold per method (cdr_threshold_per_method.csv)\n")
cat("  2. Figure: CDR x Method interaction (fig_cdr_method_interaction.pdf)\n")
cat("  3. Figure: CDR x IR x Method surface (fig_cdr_ir_interaction.pdf)\n")
cat("  4. Table: LODOCV prescriptive accuracy (lodocv_results.csv)\n")
