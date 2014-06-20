# set working directory
setwd("C:/")
if(!file.exists("./Data")) {dir.create("./Data")}

# load relavent dataset
library(caret)
library(randomForest)
library(plyr)

# download related training and testing dataset
train.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(train.url, destfile = "./Data/train.csv")
download.file(test.url, destfile = "./Data/test.csv")

# read the dataset
training <- read.csv("./Data/train.csv")
testing <- read.csv("./Data/test.csv")

# feature selections
# convert all nemeric features to numeric type
training[,6:159] <- apply(training[,6:159], 2, as.numeric)

# step 1 delete the potential features with over 90% missing values
NAs <- sapply(training, function(i) sum(is.na(i))) / nrow(training)
na.feature <- names(which(NAs > 0.9))
# a short list of deleted features
head(na.feature)
# 101 features are deleted
length(na.feature)
keep.feaure <- setdiff(names(training), na.feature)
# deleted all NA features
training <- training[,keep.feaure]

# step 2 deleted all irrelevant variables such as name, time stamp, windows..
training <- training[, 7:59]

# now perform similar treatment to testing dataset
testing <- testing[, keep.feaure[-59]]
testing <- testing[, 7:58]

# function name: sample.dataset
# df <- sample.dataset(p = 0.20, tt.num = 100, dataset = training, rand.seed = 0)
# sample p% proportion of the dataset as training set
# then use sample n of the rest of the dataset as testing set

# Input>
# p         proportion of the dataset to be sampled (0.1)
# dataset   the dataset to be splited into training and testing (training)
# rand.seed the seed use for generating random sample (0)
# tt.num the number to be sampled to training dataset (100)

sample.dataset <- function(p, tt.num, dataset, rand.seed){
    set.seed(rand.seed)
    train.idx <- createDataPartition(dataset$classe, p = p , list = F)
    train <- dataset[train.idx, ]
    test.idx <- c(1:nrow(dataset))[-train.idx]
    test <- dataset[sample(test.idx, tt.num), ]
    df <- list(train = train, test = test)
    df
}

# Random Forest
# Since we already have a 20 observations testing sets, then we only use the training set to do
# a cross validation and apply a random Forest one time, and choose the best top 5 features.

df <- sample.dataset(p = 0.2, tt.num = 200, dataset = training, rand.seed = 2014)
forest <- randomForest(classe ~., data = df$train, 
                     mtry = 2, importance = TRUE, do.trace = 100)
pred <- predict(forest, df$test)
acc <- table(pred, df$test$classe)
# confusion matrix
acc
# overall accuarcy 96.5%
sum(diag(acc)) / sum(acc)

# Then use Gini importance to select the top 5 importance features
Imp <- varImp(forest, scale = T)
Imp

# use the average Importance as a ceriteria to select features
AvgImp <- apply(Imp, 1, mean) 
AvgImp
top5 <- head(names(AvgImp[order(AvgImp, decreasing = T)]),5)
top5

# subset the training dataset with the top 5 features
training <- subset(training, select = c(top5, "classe"))
# have a look at the dimension
dim(training)

# sample a again using the same seed and process
df <- sample.dataset(p = 0.2, tt.num = 200, dataset = training, rand.seed = 2014)
forest1 <- randomForest(classe ~., data = df$train, 
                       mtry = 2, importance = TRUE, do.trace = 100)
pred1 <- predict(forest1, df$test)
acc1 <- table(pred1, df$test$classe)
# confusion matrix
acc1
# After only use the top importance features, the overall accuarcy decreases to 93.5%
sum(diag(acc1)) / sum(acc1)

# Now we explore the ovarall accuracy's relationship with the growing of training size
# First we aggregate our process above into a function
# This function will automatically do the sampling on the dataset with proportion given
# and sample 100 out of the train as testing for cross validation. Also we sample 100
# in the train dataset to show the in-sample-errors.

forest <- function(p, tt.num, dataset, rand.seed){
    tt <- proc.time()
    # sample data
    df <- sample.dataset(p, tt.num, dataset, rand.seed)    
        
    # train test
    ntrain <- nrow(df$train)
    if(ntrain <= 100){
        train.idx <- c(1:ntrain) 
    }else{
        train.idx <- sample(1:ntrain, 100)
    }
    
    # build tree
    tree <- randomForest(classe ~., data = df$train, 
                             mtry = 2, importance = TRUE, do.trace = 100)
    
    # predict values of the train data and accuracy
    tree.tr.pred <- predict(tree, newdata = df$train[train.idx,])
    tree.tr.accu = table(df$train[train.idx,]$classe, tree.tr.pred)
    
    # predict values of the test data and accuracy
    tree.tt.pred = predict(tree, newdata = df$test)
    tree.tt.accu = table(df$test$classe, tree.tt.pred)
    time <- round((proc.time() - tt)[3], 3)
    # return the result
    result <- list(train = tree.tr.accu, test = tree.tt.accu, time = time)
    message(paste("Time Takes", time))
    result
}

# we want to try the following porportion with 5 replicates.
p <- rep(c(0.05, 0.1, 0.2, 0.4, 0.8, 0.9), 5)
# set the same seed for every porportion experiment such that they will have
# the same training and testing dataset
seed <- 2014 + rep(1:5, each = length(p))

# Now use lapply to generated the 60 confusion matrix (30 for in sample 30 for out of sample).
rlist <- lapply(1:30, function(i) forest(p = p[i], tt.num = 100, dataset = training, rand.seed = seed[i]))

# now plot those accuracy to show if there is some relationship
plot.error <- function(step, rlist, title = ""){
    require(gridExtra)
    tr.acc <- sapply(rlist, function(i) sum(diag(i$train)) / sum(i$train))
    tt.acc <- sapply(rlist, function(i) sum(diag(i$test)) / sum(i$test))    
    time <- sapply(rlist, function(i) i$time)    
    
    # calculate overall accuracy
    all.tr <- data.frame(step = step, acc = tr.acc, type = "train")
    all.tt <- data.frame(step = step, acc = tt.acc, type = "test")
    all.time <- data.frame(step = step, time = time)
    all.res <- rbind(all.tr, all.tt)
    
    all.p <- ddply(all.res, .(step, type), summarize, sd = sd(acc), mean = mean(acc), se = sd(acc)/sqrt(length(acc)), count = length(acc))
    all.time.p <- ddply(all.time, .(step), summarize, sd = sd(time), mean = mean(time), se = sd(time) / sqrt(length(time)), count = length(time))
    
    p1 <- ggplot(all.p, aes(x = step*100, y = mean, colour = type, label = round(mean,2))) + 
        geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 3, lwd = 0.7) +
        geom_line(lwd = 0.7) +
        geom_point(size = 2) +
        geom_text(size = 4, color = "black", hjust = 1) +
        theme_bw() +
        ggtitle(paste0(title, " on 5 features (rept num = ", unique(all.p$count), ")")) + 
        xlab("Percentage(%) of the training") +  
        ylab("Overall Accuracy")
    
    p2 <- ggplot(all.time.p, aes(x = step*100, y = mean, color = "red", label = round(mean,2))) + 
        geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 3, lwd = 0.7) +
        geom_line(lwd = 0.7) +
        geom_point(size = 2) +
        geom_text(size = 4, color = "black", hjust = 1) +
        theme_bw() +
        theme(legend.position="none") +
        ggtitle(paste0(title, "'s Average Computational Time")) + 
        xlab("Percentage(%) of the training") +  
        ylab("Time in Second") 
    grid.arrange(p1, p2, ncol = 2)
    
}

png("randomForest.png", width = 960, height = 480)
plot.error(p, rlist, "Random Tree")
dev.off()