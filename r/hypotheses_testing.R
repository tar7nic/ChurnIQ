# ── 0. Packages ───────────────────────────────────────────────────────────────
required_packages <- c("ggplot2","dplyr","readr","tidyr","coin","rstatix","ggpubr")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
library(ggplot2); library(dplyr); library(readr)
library(tidyr);   library(coin);  library(rstatix); library(ggpubr)

# ── 1. Load & prep ────────────────────────────────────────────────────────────
DATA_PATH  <- file.path("data", "raw", "churn_raw.csv")
OUTPUT_DIR <- file.path("outputs", "figures")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(DATA_PATH, show_col_types = FALSE) %>%
  mutate(
    tenure_group = case_when(
      tenure <= 12 ~ "0-12",
      tenure <= 24 ~ "13-24",
      tenure <= 48 ~ "25-48",
      TRUE         ~ "49-72"
    ),
    Churn         = factor(ifelse(tolower(churn) %in% c("yes","1","true"),
                                  "Churned", "Retained")),
    ContractType  = as.factor(contract),
    PaymentMethod = as.factor(payment_method),
    tenure_group  = as.factor(tenure_group)
  ) %>%
  filter(!is.na(Churn))

cat(sprintf("Rows: %d | Churned: %d | Retained: %d\n",
            nrow(df), sum(df$Churn=="Churned"), sum(df$Churn=="Retained")))

# ── A. Wilcoxon Rank-Sum — Continuous Features ────────────────────────────────
continuous_vars <- c("monthly_charges","total_charges",
                     "tenure","num_support_tickets","satisfaction_score")

cat("\n── Wilcoxon Rank-Sum Tests ──\n")
wilcox_results <- lapply(continuous_vars, function(var) {
  test    <- wilcox.test(as.formula(paste(var, "~ Churn")), data = df, exact = FALSE)
  n       <- nrow(df)
  z_stat  <- qnorm(test$p.value / 2) * sign(
               mean(df[[var]][df$Churn=="Churned"],  na.rm=TRUE) -
               mean(df[[var]][df$Churn=="Retained"], na.rm=TRUE))
  r_eff   <- abs(z_stat) / sqrt(n)
  cat(sprintf("  %-25s W=%-10.1f p=%.4f r=%.3f %s\n", var, test$statistic,
              test$p.value, r_eff,
              ifelse(test$p.value<0.001,"***",ifelse(test$p.value<0.01,"**",
              ifelse(test$p.value<0.05,"*","ns")))))
  data.frame(Variable=var, W=test$statistic, P_value=test$p.value,
             Effect_r=round(r_eff,3), Significant=test$p.value<0.05)
})
wilcox_df <- bind_rows(wilcox_results)

# Violin plots
df_long <- df %>%
  select(Churn, all_of(continuous_vars)) %>%
  pivot_longer(-Churn, names_to="Variable", values_to="Value")

p_violin <- ggplot(df_long, aes(x=Churn, y=Value, fill=Churn)) +
  geom_violin(alpha=0.6, trim=FALSE) +
  geom_boxplot(width=0.15, outlier.size=0.5, fill="white", alpha=0.8) +
  facet_wrap(~Variable, scales="free_y", ncol=3) +
  scale_fill_manual(values=c("Churned"="#E74C3C","Retained"="#3498DB")) +
  labs(title="Feature Distributions: Churned vs Retained",
       x=NULL, y="Value") +
  theme_minimal(base_size=12) +
  theme(legend.position="none", strip.text=element_text(face="bold"))

ggsave(file.path(OUTPUT_DIR, "hypothesis_violin_plots.png"),
       plot=p_violin, width=12, height=8, dpi=150)
cat("Saved → hypothesis_violin_plots.png\n")

# ── B. Chi-Squared — Categorical Features ────────────────────────────────────
categorical_vars <- c("ContractType","PaymentMethod","internet_service","tenure_group")

cat("\n── Chi-Squared Tests ──\n")
chi_results <- lapply(categorical_vars, function(var) {
  tbl    <- table(df[[var]], df$Churn)
  test   <- chisq.test(tbl, simulate.p.value=TRUE, B=2000)
  n      <- sum(tbl); k <- min(nrow(tbl), ncol(tbl))
  cramer <- sqrt(test$statistic / (n * (k-1)))
  cat(sprintf("  %-25s chi2=%-8.2f p=%.4f V=%.3f %s\n", var, test$statistic,
              test$p.value, cramer,
              ifelse(test$p.value<0.001,"***",ifelse(test$p.value<0.01,"**",
              ifelse(test$p.value<0.05,"*","ns")))))
  data.frame(Variable=var, Chi2=round(test$statistic,3), P_value=test$p.value,
             CramersV=round(cramer,3), Significant=test$p.value<0.05)
})
chi_df <- bind_rows(chi_results)

# Churn rate bar plots
plots_cat <- lapply(categorical_vars, function(var) {
  df %>%
    group_by(.data[[var]], Churn) %>% summarise(n=n(), .groups="drop") %>%
    group_by(.data[[var]]) %>% mutate(pct=n/sum(n)*100) %>%
    filter(Churn=="Churned") %>%
    ggplot(aes(x=reorder(.data[[var]],-pct), y=pct, fill=.data[[var]])) +
    geom_col(alpha=0.85, width=0.6) +
    geom_text(aes(label=sprintf("%.1f%%",pct)), vjust=-0.4, size=3.5, fontface="bold") +
    scale_fill_brewer(palette="Set2") +
    labs(title=paste("Churn Rate by", var), x=var, y="Churn Rate (%)") +
    theme_minimal(base_size=12) +
    theme(legend.position="none", axis.text.x=element_text(angle=20, hjust=1))
})
p_cat <- ggarrange(plotlist=plots_cat, ncol=2, nrow=2, labels="AUTO")
ggsave(file.path(OUTPUT_DIR, "hypothesis_churn_rate_categorical.png"),
       plot=p_cat, width=13, height=9, dpi=150)
cat("Saved → hypothesis_churn_rate_categorical.png\n")

# ── C. ANOVA — satisfaction_score across contract types ──────────────────────
cat("\n── One-Way ANOVA: satisfaction_score ~ contract ──\n")
aov_model <- aov(satisfaction_score ~ ContractType, data=df)
print(summary(aov_model))
print(TukeyHSD(aov_model))

p_aov <- ggplot(df, aes(x=ContractType, y=satisfaction_score, fill=ContractType)) +
  geom_boxplot(alpha=0.7, outlier.size=0.8) +
  stat_compare_means(method="anova", label.y=max(df$satisfaction_score)+0.3) +
  scale_fill_brewer(palette="Pastel1") +
  labs(title="Satisfaction Score by Contract Type",
       subtitle="One-Way ANOVA with Tukey HSD",
       x="Contract Type", y="Satisfaction Score") +
  theme_minimal(base_size=13) + theme(legend.position="none")

ggsave(file.path(OUTPUT_DIR, "hypothesis_anova_satisfaction.png"),
       plot=p_aov, width=8, height=6, dpi=150)
cat("Saved → hypothesis_anova_satisfaction.png\n")

# ── D. Summary CSV ────────────────────────────────────────────────────────────
wilcox_df$TestType <- "Wilcoxon"; wilcox_df$Statistic <- wilcox_df$W
wilcox_df$Effect   <- paste0("r=", wilcox_df$Effect_r)
chi_df$TestType    <- "Chi-squared"; chi_df$Statistic <- chi_df$Chi2
chi_df$Effect      <- paste0("V=", chi_df$CramersV)

summary_df <- bind_rows(
  wilcox_df %>% select(Variable, TestType, Statistic, P_value, Effect, Significant),
  chi_df    %>% select(Variable, TestType, Statistic, P_value, Effect, Significant)
) %>% arrange(P_value)

write_csv(summary_df, file.path("outputs", "hypothesis_test_results.csv"))
cat("Saved → hypothesis_test_results.csv\n")

cat("\n✅ Hypothesis testing complete.\n")