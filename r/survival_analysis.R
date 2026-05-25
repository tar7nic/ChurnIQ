# ── 0. Packages ───────────────────────────────────────────────────────────────
required_packages <- c("survival", "survminer", "ggplot2", "dplyr", "readr")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
library(survival); library(survminer)
library(ggplot2);  library(dplyr); library(readr)

# ── 1. Load ───────────────────────────────────────────────────────────────────
DATA_PATH  <- file.path("data", "raw", "churn_raw.csv")
OUTPUT_DIR <- file.path("outputs", "figures")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(DATA_PATH, show_col_types = FALSE)

# ── 2. Create tenure_group + survival columns ─────────────────────────────────
df <- df %>%
  mutate(
    tenure_group = case_when(
      tenure <= 12 ~ "0-12",
      tenure <= 24 ~ "13-24",
      tenure <= 48 ~ "25-48",
      TRUE         ~ "49-72"
    ),
    surv_time    = as.numeric(tenure),
    surv_status  = ifelse(tolower(churn) %in% c("yes","1","true"), 1L, 0L),
    ContractType  = as.factor(contract),
    TenureGroup   = as.factor(tenure_group),
    PaymentMethod = as.factor(payment_method)
  ) %>%
  filter(!is.na(surv_time), surv_time > 0)

cat(sprintf("Rows: %d | Churn events: %d\n", nrow(df), sum(df$surv_status)))

# ── 3. KM by Contract Type ────────────────────────────────────────────────────
km_fit_contract <- survfit(Surv(surv_time, surv_status) ~ ContractType, data = df)

p_km_contract <- ggsurvplot(
  km_fit_contract, data = df,
  pval = TRUE, conf.int = TRUE, risk.table = TRUE, risk.table.height = 0.28,
  legend.title = "Contract Type",
  xlab = "Tenure (Months)", ylab = "Survival Probability",
  title = "Kaplan-Meier Survival Curves by Contract Type",
  palette = c("#E74C3C", "#3498DB", "#2ECC71"),
  ggtheme = theme_minimal(base_size = 13)
)
png(file.path(OUTPUT_DIR, "km_survival_contract.png"), width = 1500, height = 1050, res = 150)
print(p_km_contract)
dev.off()
cat("Saved → km_survival_contract.png\n")

# ── 4. KM by Tenure Group ─────────────────────────────────────────────────────
km_fit_tenure <- survfit(Surv(surv_time, surv_status) ~ TenureGroup, data = df)

p_km_tenure <- ggsurvplot(
  km_fit_tenure, data = df,
  pval = TRUE, conf.int = FALSE, risk.table = TRUE, risk.table.height = 0.30,
  legend.title = "Tenure Group",
  xlab = "Tenure (Months)", ylab = "Survival Probability",
  title = "Kaplan-Meier Survival Curves by Tenure Group",
  palette = "Dark2", ggtheme = theme_minimal(base_size = 13)
)
png(file.path(OUTPUT_DIR, "km_survival_tenure_group.png"), width = 1500, height = 1050, res = 150)
print(p_km_tenure)
dev.off()
cat("Saved → km_survival_tenure_group.png\n")

# ── 5. Log-rank test ──────────────────────────────────────────────────────────
logrank_test <- survdiff(Surv(surv_time, surv_status) ~ ContractType, data = df)
cat("\n── Log-Rank Test (Contract Type) ──\n"); print(logrank_test)

# ── 6. Cox PH Model ───────────────────────────────────────────────────────────
cox_model <- coxph(
  Surv(surv_time, surv_status) ~
    monthly_charges + num_support_tickets + satisfaction_score +
    ContractType + PaymentMethod,
  data = df, ties = "efron"
)
cat("\n── Cox PH Summary ──\n"); print(summary(cox_model))

# ── 7. PH Assumption Check ────────────────────────────────────────────────────
ph_test <- cox.zph(cox_model)
cat("\n── Schoenfeld Residuals ──\n"); print(ph_test)

png(file.path(OUTPUT_DIR, "cox_ph_schoenfeld.png"), width = 1200, height = 800, res = 150)
  ggcoxzph(ph_test)
dev.off()
cat("Saved → cox_ph_schoenfeld.png\n")

# ── 8. Hazard Ratio Forest Plot ───────────────────────────────────────────────
p_hr <- ggforest(cox_model, data = df,
                 main = "Hazard Ratios — Cox PH Model\n(HR > 1 = higher churn risk)",
                 cpositions = c(0.02, 0.22, 0.4), fontsize = 0.9)
ggsave(file.path(OUTPUT_DIR, "cox_hazard_ratios.png"),
       plot = p_hr, width = 10, height = 6, dpi = 150)
cat("Saved → cox_hazard_ratios.png\n")

# ── 9. Median survival time ───────────────────────────────────────────────────
cat("\n── Median Survival by Contract Type ──\n"); print(km_fit_contract)

# ── 10. Save Cox coefficients ─────────────────────────────────────────────────
cox_coef_df <- as.data.frame(summary(cox_model)$coefficients)
cox_coef_df$Variable <- rownames(cox_coef_df); rownames(cox_coef_df) <- NULL
cox_coef_df <- cox_coef_df[, c("Variable","coef","exp(coef)","se(coef)","z","Pr(>|z|)")]
colnames(cox_coef_df) <- c("Variable","Coef","HazardRatio","SE","Z","PValue")
write_csv(cox_coef_df, file.path("outputs", "cox_model_coefficients.csv"))
cat("Saved → cox_model_coefficients.csv\n")

cat("\n✅ Survival analysis complete.\n")