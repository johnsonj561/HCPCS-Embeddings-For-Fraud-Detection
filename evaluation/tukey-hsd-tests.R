library(agricolae)

results_path = '~/git/HCPCS-Embeddings-For-Fraud-Detection/xgboost/test-results.csv'
results = read.csv(results_path, header=TRUE, sep=',')


# AUC ANOVA + HSD
anova_results <- aov(roc_auc ~embedding_type, data=results)
summary(anova_results)
tukey_results = HSD.test(anova_results, "embedding_type", group=TRUE, alpha=0.1)
print('AUC Mean HSD Groups')
tukey_results

library(gplots)
warnings()
# Plot the mean of teeth length by dose groups
plotmeans(roc_auc~embedding_type, data = results, frame = TRUE, connect = FALSE, p=0.99, n.label = FALSE, )

