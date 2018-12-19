library(tidyverse)
library(ggsignif)
library(lsmeans)
setwd("C:/Users/federico nemmi/Documents/Python Scripts/dystacmap/")

dat <- read_csv("bootstrap_results.csv") %>%
  select(-X1) %>%
  gather()

dat$key <- as.factor(dat$key)
contrasts(dat$key) <- contr.sum(8)

mod <- lm(value ~ key, dat)
anova(mod)
lsmeans(mod, pairwise ~ key, adjust = "bonferroni")


dat <- read_csv("bootstrap_results_100_reps.csv") %>%
  select(-X1) %>%
  gather()

dat$key <- as.factor(dat$key)
contrasts(dat$key) <- contr.sum(8)

mod <- lm(value ~ key, dat)
anova(mod)
lsmeans(mod, pairwise ~ key, adjust = "bonferroni")


tiff("discrimination_comparison.tiff", w = 2055, h = 2055, res = 600)
ggplot(dat %>% group_by(key) %>%
         summarise(avg = mean(value),
                   lower_ci = t.test(value)$conf.int[1],
                   upper_ci = t.test(value)$conf.int[2]) %>%
         mutate(key = factor(key, levels = c("gm", "wm", "structural", "fALFF", "localCorr", "globalCorr",
                                              "functional", "complete"))),
       aes(x = key, y = avg)) + 
  geom_bar(stat = "identity", fill = "dodgerblue", width = .6) + 
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = .2) + 
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14)) +
  scale_y_continuous(limits = c(0,1)) + 
  geom_signif(xmin = 1, xmax = 2, y = .55, annotation = "NS") +
  geom_signif(xmin = 3, xmax = 5, y = .95, annotation = "NS") + 
  geom_signif(xmin = 7, xmax = 8, y = .85, annotation = "NS") +
  geom_signif(xmin = 4, xmax = 6, y = .75, annotation = "NS") +
  geom_signif(xmin = 4, xmax = 5, y = .85, annotation = "NS") +
  geom_signif(xmin = 5, xmax = 6, y = .85, annotation = "NS") +
  geom_signif(xmin = 1, xmax = 3, y = .65, annotation = "NS") +
  xlab("Modalities used") + 
  ylab("Balanced Accuracy")
dev.off()
