#!/usr/bin/env Rscript
# ═══════════════════════════════════════════════════════════════════
# Model Diagnostics for LME Model (Critical #3)
# - QQ plot of residuals
# - Residual vs fitted
# - ICC
# - VIF for collinearity (especially log_CDR vs log_K)
# - Nested model comparison (null vs main vs full)
# ═══════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(ggplot2)
  library(MuMIn)
  library(car)
})

INPUT  <- "meta_dataset_kd_only.csv"
OUTDIR <- "stage2_output"

df <- read.csv(INPUT, stringsAsFactors = FALSE) %>%
  filter(method != "label_smoothing") %>%
  filter(!is.na(delta_f1))

df$method  <- factor(df$method)
df$family  <- factor(df$family)
df$dataset <- factor(df$dataset)
df$student <- factor(df$student)
df$method  <- relevel(df$method, ref = "vanilla")

cat("═══════════════════════════════════════════════════\n")
cat("Model Diagnostics\n")
cat("═══════════════════════════════════════════════════\n\n")

# ─── Fit models for comparison ──────────────────────────────────
cat("1. Fitting nested models for comparison...\n")

m_null <- lmer(delta_f1 ~ 1 + (1 | dataset) + (1 | student),
               data = df, REML = FALSE,
               control = lmerControl(optimizer = "bobyqa"))

m_main <- lmer(delta_f1 ~ log_CDR + log_IR + baseline_f1 + log_K + method +
                 (1 | dataset) + (1 | student),
               data = df, REML = FALSE,
               control = lmerControl(optimizer = "bobyqa"))

m_full <- lmer(delta_f1 ~ log_CDR + log_IR + baseline_f1 + log_K + method +
                 log_CDR:method + log_IR:method + log_CDR:log_IR +
                 baseline_f1:method +
                 (1 | dataset) + (1 | student),
               data = df, REML = FALSE,
               control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))

cat("\n   Model comparison:\n")
cat(sprintf("   %-20s AIC=%8.1f  BIC=%8.1f  R²m=%.4f  R²c=%.4f\n",
            "Null (intercept)",
            AIC(m_null), BIC(m_null),
            r.squaredGLMM(m_null)[1,"R2m"], r.squaredGLMM(m_null)[1,"R2c"]))
cat(sprintf("   %-20s AIC=%8.1f  BIC=%8.1f  R²m=%.4f  R²c=%.4f\n",
            "Main effects",
            AIC(m_main), BIC(m_main),
            r.squaredGLMM(m_main)[1,"R2m"], r.squaredGLMM(m_main)[1,"R2c"]))
cat(sprintf("   %-20s AIC=%8.1f  BIC=%8.1f  R²m=%.4f  R²c=%.4f\n",
            "Full (interactions)",
            AIC(m_full), BIC(m_full),
            r.squaredGLMM(m_full)[1,"R2m"], r.squaredGLMM(m_full)[1,"R2c"]))

# LRT: null vs main
lrt_nm <- anova(m_null, m_main)
cat(sprintf("\n   LRT null vs main: χ²=%.1f, df=%d, p=%.2e\n",
            lrt_nm$Chisq[2], lrt_nm$`Chi Df`[2], lrt_nm$`Pr(>Chisq)`[2]))

# LRT: main vs full
lrt_mf <- anova(m_main, m_full)
cat(sprintf("   LRT main vs full: χ²=%.1f, df=%d, p=%.2e\n",
            lrt_mf$Chisq[2], lrt_mf$`Chi Df`[2], lrt_mf$`Pr(>Chisq)`[2]))

# ─── 2. ICC ────────────────────────────────────────────────────
cat("\n2. Intraclass Correlation Coefficients (from null model)...\n")
vc_null <- as.data.frame(VarCorr(m_null))
total_var <- sum(vc_null$vcov)
cat(sprintf("   ICC(dataset) = %.4f  (%.1f%%)\n",
            vc_null$vcov[1]/total_var, 100*vc_null$vcov[1]/total_var))
cat(sprintf("   ICC(student) = %.4f  (%.1f%%)\n",
            vc_null$vcov[2]/total_var, 100*vc_null$vcov[2]/total_var))
cat(sprintf("   Residual     = %.4f  (%.1f%%)\n",
            vc_null$vcov[3]/total_var, 100*vc_null$vcov[3]/total_var))

# ─── 3. VIF (on fixed-effects-only model) ─────────────────────
cat("\n3. Variance Inflation Factors...\n")
m_lm <- lm(delta_f1 ~ log_CDR + log_IR + baseline_f1 + log_K + method,
            data = df)
vif_vals <- vif(m_lm)
cat("   (Using linear model for VIF calculation)\n")
print(vif_vals)

cat(sprintf("\n   Correlation between log_CDR and log_K: r = %.4f\n",
            cor(df$log_CDR, df$log_K)))

# ─── 4. Residual diagnostics ──────────────────────────────────
cat("\n4. Residual diagnostics...\n")
resids <- residuals(m_full)
fitted_vals <- fitted(m_full)

cat(sprintf("   Shapiro-Wilk (on sample of 5000): "))
set.seed(42)
sw_sample <- sample(resids, min(5000, length(resids)))
sw_test <- shapiro.test(sw_sample)
cat(sprintf("W=%.4f, p=%.2e\n", sw_test$statistic, sw_test$p.value))

cat(sprintf("   Skewness: %.4f\n", moments::skewness(resids)))
cat(sprintf("   Kurtosis: %.4f\n", moments::kurtosis(resids)))

# QQ plot
pdf(file.path(OUTDIR, "figures", "fig_qq_residuals.pdf"), width = 5, height = 4.5)
qqnorm(resids, main = "Normal Q-Q Plot of LME Residuals",
       pch = 16, cex = 0.3, col = rgb(0,0,0,0.2))
qqline(resids, col = "red", lwd = 1.5)
dev.off()
cat("   fig_qq_residuals.pdf saved.\n")

# Residual vs fitted
pdf(file.path(OUTDIR, "figures", "fig_residual_vs_fitted.pdf"), width = 6, height = 4.5)
plot(fitted_vals, resids, pch = 16, cex = 0.3, col = rgb(0,0,0,0.15),
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs. Fitted Values")
abline(h = 0, col = "red", lwd = 1.5, lty = 2)
lo <- loess(resids ~ fitted_vals, span = 0.3)
ord <- order(fitted_vals)
lines(fitted_vals[ord], predict(lo)[ord], col = "blue", lwd = 1.5)
dev.off()
cat("   fig_residual_vs_fitted.pdf saved.\n")

# ─── 5. Save diagnostics summary ──────────────────────────────
sink(file.path(OUTDIR, "tables", "model_diagnostics.txt"))
cat("Model Diagnostics Summary\n")
cat("═════════════════════════\n\n")

cat("1. Nested Model Comparison:\n")
cat(sprintf("   %-20s AIC=%8.1f  BIC=%8.1f  R²m=%.4f  R²c=%.4f\n",
            "Null", AIC(m_null), BIC(m_null),
            r.squaredGLMM(m_null)[1,"R2m"], r.squaredGLMM(m_null)[1,"R2c"]))
cat(sprintf("   %-20s AIC=%8.1f  BIC=%8.1f  R²m=%.4f  R²c=%.4f\n",
            "Main effects", AIC(m_main), BIC(m_main),
            r.squaredGLMM(m_main)[1,"R2m"], r.squaredGLMM(m_main)[1,"R2c"]))
cat(sprintf("   %-20s AIC=%8.1f  BIC=%8.1f  R²m=%.4f  R²c=%.4f\n",
            "Full", AIC(m_full), BIC(m_full),
            r.squaredGLMM(m_full)[1,"R2m"], r.squaredGLMM(m_full)[1,"R2c"]))

cat(sprintf("\n   LRT null vs main: χ²=%.1f, df=%d, p=%.2e\n",
            lrt_nm$Chisq[2], lrt_nm$`Chi Df`[2], lrt_nm$`Pr(>Chisq)`[2]))
cat(sprintf("   LRT main vs full: χ²=%.1f, df=%d, p=%.2e\n",
            lrt_mf$Chisq[2], lrt_mf$`Chi Df`[2], lrt_mf$`Pr(>Chisq)`[2]))

cat("\n2. ICC (null model):\n")
cat(sprintf("   ICC(dataset) = %.4f\n", vc_null$vcov[1]/total_var))
cat(sprintf("   ICC(student) = %.4f\n", vc_null$vcov[2]/total_var))

cat("\n3. VIF:\n")
print(vif_vals)
cat(sprintf("\n   r(log_CDR, log_K) = %.4f\n", cor(df$log_CDR, df$log_K)))

cat("\n4. Residual diagnostics:\n")
cat(sprintf("   Shapiro-Wilk: W=%.4f, p=%.2e\n", sw_test$statistic, sw_test$p.value))

cat("\n5. Random effects variance (full model):\n")
print(VarCorr(m_full))

sink()
cat("\n   model_diagnostics.txt saved.\n")

cat("\n═══════════════════════════════════════════════════\n")
cat("DIAGNOSTICS COMPLETE\n")
cat("═══════════════════════════════════════════════════\n")
