setwd("N:/dan/labelPrediction/r") 
library(ggplot2)

fileName = "facebook-allLabels"

data <- read.table(file = paste('../data/', fileName , '.txt' , sep = ""), header = T, sep="\t")
#ggplot(res, aes(x,fill=name)) + geom_histogram(position = 'dodge') + theme_bw()
ggsave(file=paste('./plots/', fileName , '.png' , sep = ""), width=5, height=5)

d1 <- data.frame(x=data$Sex)
d2 <- data.frame(x=data$Political.View)
d3 <- data.frame(x=data$Religious.View)

d1$name <- 'Sex'
d2$name <- 'Political.View'
d3$name <- 'Religious.View'

res <- rbind(d1,d2,d3)

ggplot(res, aes(x,fill=name)) + geom_histogram(binwidth=0.1,position = 'dodge') + theme_bw()
ggsave(file=paste('./plots/', fileName , '.png' , sep = ""), width=5, height=5)






path = "../results/"
titleName = "politics"
titleName = "sex"
fileName = paste("basicModels_", titleName ,"",sep="")
yLabel = "recall"

data <- read.table(file = paste(path, fileName , '.txt' , sep = ""), header = T)

x <- 6
d1 <- data[,c(1,x,x+5,x+10,x+15)]
d2 <- reshape(d1,direction="long",idvar="trainingSize",varying=list(2:5),v.names = "accuracy")
d2[,2] <- factor(d2[,2])
#levels(d2[,2]) <- c("NodeFeatures","NodeFatures + UserPosting","NodeFeatures + WallPosted","NodeFeatures + UserPosting + WallPosted")
levels(d2[,2]) <- c("NodeFeatures","NF + UserPosting","NF + WallPosted","NF + UP + WP")

pd <- position_dodge(0)
ggplot(d2, aes(x=trainingSize, y=accuracy, color=time)) + 
  geom_line(position = pd) +
  geom_point(position = pd) +
  ggtitle( fileName ) +
  ylab(yLabel) 

suffix = paste(yLabel,"")
ggsave(file=paste('./plots/', fileName, '_', suffix, '.png' , sep = ""),width=9.69,height=7.79)

