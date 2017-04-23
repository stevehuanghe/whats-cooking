#set workspace path
setwd("/Users/ellen/Documents/CS412_ML/project")

library(jsonlite)
df <- fromJSON("./vtest.json", flatten = TRUE)
df4 <- data.frame(lapply(df4, as.character), stringsAsFactors=FALSE)
train <- fromJSON("./train.json", flatten = TRUE)
freq = as.data.frame(table(train$ingredients))
## load packages
library(igraph)
library(plyr)
library(ggplot2)
library(tm)
library(lsa)
g = graph.data.frame(train[,2:3], directed = T)
mat = get.adjacency(g)
mat = as.matrix(mat)
m2 = t(mat) %*% mat
title.idx = which(colSums(m2) > 0)
title.mat = m2[title.idx, title.idx]
diag(title.mat) = 0  ## co-star with self does not count
title.idx = which(colSums(title.mat) > 0)
title.mat = title.mat[title.idx, title.idx]
dim(title.mat)
title.mat[1:3,]
title.mat[which(title.mat < 3)] = 0
rownames(title.mat)[order(colSums(title.mat), decreasing = T)[1:10]]
g = graph.adjacency(title.mat, weighted = T, mode = "undirected", diag = F)
set.seed(1)
plot(g, layout = layout.fruchterman.reingold, vertex.label = V(g)$name)

plot(g, layout = layout.fruchterman.reingold,vertex.size = 8, vertex.label.cex = 0.75)
fc = fastgreedy.community(g)
modularity(fc)
membership(fc)
set.seed(1)
plot(fc, g, main = "modularity community",
     layout = layout.fruchterman.reingold,
     vertex.size = 8, vertex.label.cex = 0.5)


#tf-idf, ingrident and cusine matrix
library(tm)
ingredients <- Corpus(VectorSource(test5$ingredients))
ingredients

ingredients <- tm_map(ingredients,content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),mc.cores=1)

dt.mat = as.matrix(DocumentTermMatrix(ingredients))
td.mat = as.matrix(TermDocumentMatrix(ingredients))
dim(td.mat)




freq2 <- colSums(as.matrix(dt.mat))   
length(freq2)  

ord <- order(freq2)  
dt.mat <- removeSparseTerms(dt.mat, 0.99) # This makes a matrix that is 10% empty space, maximum.   
#inspect(dtms) 
freq2[head(ord)]  
freq2[tail(ord)]  
head(table(freq2), 20) 
tail(table(freq2), 20)   
#freq <- colSums(as.matrix(dtms))   
#freq   
freq2 <- sort(colSums(as.matrix(dt.mat)), decreasing=TRUE)   
head(freq2, 20) 
findFreqTerms(dt.mat, lowfreq=1000) 
wf <- data.frame(word=names(freq), freq=freq2)   
head(wf) 

library(ggplot2)   
p <- ggplot(subset(wf, count>1500), aes(word, freq))    
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
p   
text(x = p, y = wf$freq, label = wf$freq, pos = 3, cex = 0.8, col = "red")
findAssocs(dtm, "food", corlimit =0.1)

library(wordcloud) 
set.seed(142)   
wordcloud(names(freq), freq, min.freq=1000, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))  

dist.mat = dist(t(as.matrix(td.mat)))
doc.mds = cmdscale(dist.mat, k = 2)
## tfidf
td.mat.w <- lw_tf(td.mat) * gw_idf(td.mat)  ## tf-idf weighting
k = 4
S = svd(as.matrix(td.mat.w), nu = k, nv = k)
u = S$u
s = S$d
v = S$v
td.mat.svd = S$u %*% diag(S$d[1:k]) %*% t(S$v)
dist.mat = dist(t(td.mat.svd))
doc.mds = cmdscale(dist.mat, k = 2)
data = data.frame(x = doc.mds[, 1], y = doc.mds[, 2], topic = df$cuisine, 
                  id = row.names(df))
ggplot(data, aes(x = x, y = y, color = topic)) + geom_point()

g = graph.adjacency(data, weighted = T, mode = "undirected", diag = F)
set.seed(1)
plot(g, layout = layout.fruchterman.reingold, vertex.label = V(g)$name)



###creat dictionary for cuisine 
rownames(dt.mat) <- df$id
library(reshape2)
df3 <- melt(dt.mat)
df3<-df3[!(df3$value==0),]

g = graph.data.frame(df[,2:3], directed = T)
mat = get.adjacency(g)
mat = as.matrix(mat)
m2 = t(mat) %*% mat
ingre.idx = which(colSums(m2) > 0)
ingre.mat = m2[ingre.idx, ingre.idx]
diag(ingre.mat) = 0  ## co-star with self does not count
ingre.idx = which(colSums(ingre.mat) > 0)
ingre.mat = ingre.mat[ingre.idx, ingre.idx]
dim(ingre.mat)

ingre.mat[1:3,]
test_data = data.frame(x = )


train <- read.csv(file="./train.csv",head=T,sep=",")
gsub("c", "", train)

train$ingredients = as.character(train$ingredients)
write.csv(train, file = "./train.csv")
train_ingredients <- data.frame(lapply(train, as.character), stringsAsFactors=FALSE)
s <- strsplit(train_ingredients$ingredients, split = ",")
test1 <- data.frame(id = rep(train_ingredients$id, sapply(s, length)), ingredients = unlist(s))
test2 <- data.frame(id = rep(train_ingredients$cuisine, sapply(s, length)), ingredients = unlist(s))
ingredients <- cbind(test1,test2)
ingredients <- ingredients[,1:3]
colnames(ingredients) <- c("id", "ingredients","cuisine")
summary(ingredients)
ingredients$cuisine = as.factor(ingredients$cuisine)
test3 = afreq[which(afreq$count>100),]

test4 = asia[which(asia$ingredients == test3$ingredients),]
test5 = merge(test3,asia)
chinese = ingredients[which(ingredients$cuisine == "chinese"),]
japanese = ingredients[which(ingredients$cuisine == "japanese"),]
korean = ingredients[which(ingredients$cuisine == "korean"),]
asia = rbind(chinese,korean,japanese)
afreq = as.data.frame(table(asia$ingredients))


ggplot(data = test5, aes(x = ingredients, fill = cuisine)) + geom_histogram(binwidth = 1) + facet_grid(cuisine ~ .) + theme_bw()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5),legend.position="none")

test3 = afreq[which(ifreq$count>200),]

test4 = asia[which(asia$ingredients == test3$ingredients),]
test5 = merge(test3,ita)
italian = ingredients[which(ingredients$cuisine == "italian"),]
spanish = ingredients[which(ingredients$cuisine == "spanish"),]
greek = ingredients[which(ingredients$cuisine == "greek"),]
moroccan = ingredients[which(ingredients$cuisine == "moroccan"),]
mexican = ingredients[which(ingredients$cuisine == "mexican"),]
vietnamese = ingredients[which(ingredients$cuisine == "vietnamese"),]
thai = ingredients[which(ingredients$cuisine == "thai"),]

ita = rbind(vietnamese,thai)
ifreq = as.data.frame(table(ita$ingredients))
colnames(ifreq) <- c("ingredients", "count")



ggplot(data = test5, aes(x = ingredients, fill = cuisine)) + geom_histogram(binwidth = 1) + facet_grid(cuisine ~ .) + theme_bw()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5),legend.position="none")






train <- within(train, cuisine <- factor(cuisine, levels=names(sort(table(cuisine), decreasing=TRUE))))

g1 = ggplot(train, aes(x = cuisine,fill = cuisine)) + geom_histogram(binwidth = 1.5) + theme_bw()+theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5),legend.position="none")+ geom_text(stat='bin',aes(label=..count..),vjust=-1)+ ggtitle("Cuisine histgram")
g1

freq = as.data.frame(table(ingredients$ingredients))
freq_cuisine = as.data.frame(table(ingredients$cuisine))

colnames(afreq) <- c("ingredients", "count")
g2 = ggplot(subset(freq,freq$count>1000), 
            aes(x = ingredients,y = count, fill = ingredients)) + geom_bar(stat="identity") +
            geom_histogram(binwidth = 0.5) + theme_bw() +theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5),legend.position="none")+ geom_text(stat='identity',aes(label=count),vjust=-1,position = position_dodge(width=1.8),  size=3)+ ggtitle("Ingredients histgram (frequency more than 1000)")
g2

test5$ingredients = as.numeric(test5$ingredients)
test5$cuisine = as.numeric(test5$cuisine)
g3 =  ggplot(test5, aes(ingredients, cuisine)) + geom_boxplot(fill = "red")+
  scale_y_continuous("Item Outlet Sales")+
  labs(title = "Box Plot", x = "Outlet Identifier")
g3


italian = ingredients[which(ingredients$cuisine == "italian"),]
italian <- Corpus(VectorSource(italian$ingredients))
italian

italian <- tm_map(italian,content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),mc.cores=1)
# remove numbers
italian = tm_map(italian, removeNumbers,lazy=TRUE)

# remove stopwords
italian = tm_map(italian, function(x) removeWords(x, stopwords("english")),lazy=TRUE)

# stemming
italian = tm_map(italian, stemDocument, language = "english",lazy=TRUE)

dt.mat = as.matrix(DocumentTermMatrix(italian))
td.mat = as.matrix(TermDocumentMatrix(italian))
dim(td.mat)
freq2 <- sort(colSums(as.matrix(dt.mat)), decreasing=TRUE)  
wf <- data.frame(word=names(freq2), freq=freq2)

td.mat = TermDocumentMatrix(italian)
td.mat2 <- removeSparseTerms(td.mat, .992)
dim(td.mat2)

td.mat[5:10,1:20]
termDocMatrix <- as.matrix(td.mat2)
# change it to a Boolean matrix
termDocMatrix[termDocMatrix>=1] <- 1
# transform into a term-term adjacency matrix
termMatrix <- termDocMatrix %*% t(termDocMatrix)
# inspect terms numbered 5 to 10
termMatrix[5:10,5:10]
library(igraph)
# build a graph from the above matrix
g <- graph.adjacency(termMatrix, weighted=T, mode = "undirected")
# remove loops
g <- simplify(g)
# set labels and degrees of vertices
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
plot(g, layout=layout1)
V(g)$label.cex <- 2.2 * V(g)$degree / max(V(g)$degree)+ .2
V(g)$label.color <- rgb(0, 0, .2, .8)
V(g)$frame.color <- NA
egam <- (log(E(g)$weight)+.4) / max(log(E(g)$weight)+.4)
E(g)$color <- rgb(.5, .5, 0, egam)
E(g)$width <- egam
# plot the graph in layout1
plot(g, layout=layout1)
title("Italian ingredients network")
fc = fastgreedy.community(g)
modularity(fc)
membership(fc)
set.seed(1)
plot(fc, g, main = "modularity community for italian cuisine",
     layout = layout.fruchterman.reingold,
     vertex.size = 8, vertex.label.cex = 0.5)

dendPlot(fc)
deg=degree(g)
deg
top = order(deg, decreasing=T)[1:10]
top1 = order(deg, decreasing=T)[1:1]
top2 = order(deg, decreasing=T)[2:2]
top3 = order(deg, decreasing=T)[3:3]
top4 = order(deg, decreasing=T)[4:4]
top5 = order(deg, decreasing=T)[5:5]

V(g)$size = abs(deg) * 0.8
V(g)$color = "white"
V(g)$label.color = "gray33"
V(g)$label.cex = 0.66
E(g)$color = "black"
V(g)[top]$label.color = "black"  ## highlight the top-5 nodes
V(g)[top]$label.cex = 1
V(g)[top1]$color = "red"
V(g)[top2]$color = "yellow1"
V(g)[top3]$color = "green1"
V(g)[top4]$color = "deepskyblue"
V(g)[top5]$color = "mediumpurple1"

set.seed(1)
plot(g, layout = layout.circle)
title("degree centrality for italian cuisine")

clo = closeness(g)
clo
top = order(clo, decreasing=T)[1:10]
top1 = order(clo, decreasing=T)[1:1]
top2 = order(clo, decreasing=T)[2:2]
top3 = order(clo, decreasing=T)[3:3]
top4 = order(clo, decreasing=T)[4:4]
top5 = order(clo, decreasing=T)[5:5]

V(g)$size = (abs(clo)) * 60000
V(g)$color = "white"
V(g)$label.color = "gray33"
V(g)$label.cex = 0.66
V(g)[top]$label.color = "black"  ## highlight the top-5 nodes
V(g)[top1]$color = "red"
V(g)[top2]$color = "yellow1"
V(g)[top3]$color = "green1"
V(g)[top4]$color = "deepskyblue"
V(g)[top5]$color = "mediumpurple1"
V(g)[top]$label.cex = 1
set.seed(1)
plot(g, layout = layout.circle)
title("closeness for italian cuisine")

bet = betweenness(g)
bet

top = order(bet, decreasing=T)[1:10]
top1 = order(bet, decreasing=T)[1:1]
top2 = order(bet, decreasing=T)[2:2]
top3 = order(bet, decreasing=T)[3:3]
top4 = order(bet, decreasing=T)[4:4]
top5 = order(bet, decreasing=T)[5:5]

V(g)$size = abs(bet) * 0.1
V(g)$color = "white"
V(g)$label.color = "gray33"
V(g)$label.cex = 0.66
V(g)[top]$label.color = "black"  ## highlight the top-5 nodes
V(g)[top1]$color = "red"
V(g)[top2]$color = "yellow1"
V(g)[top3]$color = "green1"
V(g)[top4]$color = "deepskyblue"
V(g)[top5]$color = "mediumpurple1"
V(g)[top]$label.cex = 1
set.seed(1)
plot(g, layout = layout.circle)
title("betweenness for italian cuisine")


#italian = ingredients[which(ingredients$cuisine == "italian"),]
chinese <- Corpus(VectorSource(chinese$ingredients))
chinese

chinese <- tm_map(chinese,content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),mc.cores=1)
# remove numbers
chinese = tm_map(chinese, removeNumbers,lazy=TRUE)

# remove stopwords
chinese = tm_map(chinese, function(x) removeWords(x, stopwords("english")),lazy=TRUE)

# stemming
chinese = tm_map(chinese, stemDocument, language = "english",lazy=TRUE)

dt.matc = as.matrix(DocumentTermMatrix(chinese))
td.matc = as.matrix(TermDocumentMatrix(chinese))
dim(td.matc)
freq2 <- sort(colSums(as.matrix(dt.mat)), decreasing=TRUE)  
wf <- data.frame(word=names(freq2), freq=freq2)

td.matc = TermDocumentMatrix(chinese)
td.mat3 <- removeSparseTerms(td.matc, .992)
dim(td.mat3)

td.mat[5:10,1:20]
termDocMatrix <- as.matrix(td.mat3)
# change it to a Boolean matrix
termDocMatrix[termDocMatrix>=1] <- 1
# transform into a term-term adjacency matrix
termMatrix <- termDocMatrix %*% t(termDocMatrix)
# inspect terms numbered 5 to 10
termMatrix[5:10,5:10]
library(igraph)
# build a graph from the above matrix
g <- graph.adjacency(termMatrix, weighted=T, mode = "undirected")
# remove loops
g <- simplify(g)
# set labels and degrees of vertices
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
plot(g, layout=layout1)
V(g)$label.cex <- 2.2 * V(g)$degree / max(V(g)$degree)+ .2
V(g)$label.color <- rgb(0, 0, .2, .8)
V(g)$frame.color <- NA
egam <- (log(E(g)$weight)+.4) / max(log(E(g)$weight)+.4)
E(g)$color <- rgb(.5, .5, 0, egam)
E(g)$width <- egam
# plot the graph in layout1
plot(g, layout=layout1)
title("Chinese ingredients network")

fc = fastgreedy.community(g)
modularity(fc)
membership(fc)
set.seed(1)
plot(fc, g, main = "modularity community for chinese cusine",
     layout = layout.fruchterman.reingold,
     vertex.size = 8, vertex.label.cex = 0.5)
