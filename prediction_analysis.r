library(ggplot2)
library(tidyverse)

source("config.r")

#TODO: make text bigger
#TODO: change the colour of the 95th percentile lines so that they are distinct
val_data <- read.table(paste0(data_dir, "LSTM_onehot_20210601-164558_validation_tar_pred.csv"), header=TRUE, sep=",")

val_data$resid <- val_data$Actual - val_data$Pred

# fit the observed and predicted so we can see how far it is from expected
model <- lm(Pred ~ Actual, data = val_data)
intercept <- as.numeric(model$coef[1])
slope <- as.numeric(model$coef[2])
r2 <- summary(model)$r.squared

# 95 percentile
bounds95CI <- quantile(val_data$resid, probs = c(0.025, 0.975))
# 99 percentile
bounds99CI <- quantile(val_data$resid, probs = c(0.005, 0.995))

interval <- bounds95CI["97.5%"] - bounds95CI["2.5%"]
m <- mean(val_data$resid)
med <- median(val_data$resid)

ggplot(val_data, aes(x=Actual, y=Pred)) +
  geom_point(aes(colour = cut(resid, c(-Inf, bounds95CI["2.5%"], bounds95CI["97.5%"], Inf))), size = 0.5) +
  scale_colour_manual(name = "resid",
                      values = c("#CA0020", "#0571B0", "#CA0020"),
                      guide = FALSE) +
  geom_line(aes(y = Actual - bounds95CI["2.5%"]), size=0.5, colour = "black") +
  geom_line(aes(y = Actual - bounds95CI["97.5%"]), size=0.5, colour = "black") +
  #geom_line(aes(y = Actual + bounds99CI["0.5%"]), size=0.5, colour = "black") +
  #geom_line(aes(y = Actual + bounds99CI["99.5%"]), size=0.5, colour = "black") +
  geom_line(aes(y = Actual), size=0.75, colour = "black") +
  geom_smooth(method = lm, se = FALSE, colour = "#1FCC00", size=0.75) +
  #ylim(-5000,12000) +
  #xlim(-5000,12000) +
  xlab("Observed RT (min)") +
  ylab("Predicted RT (min)") +
  geom_text(x=75, y=20, label=paste("R^2:", r2, "\nInterval(min):", interval, "\nInterval(min):", interval, sep=" "), size=4) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle("Actual vs Pred data. CNN")


ggsave(paste0(plot_dir, "LSTM_onehot_20210601-164558_validation_tar_pred.png"))