library(keras)
library(tidyverse)
library(MLmetrics)


modell %>% evaluate(test_images, test_labels)

DNAs <- read.csv("DNA_never_seen.csv")%>% select(-X) %>% as.matrix()
Proteins <- read.csv("Protein_never_seen.csv") %>% select(-X)%>% as.matrix()
Ys <- read.csv("y_never_seen.csv") %>% select(x)%>% as.matrix()

modell <- load_model_hdf5("25.h5", custom_objects = NULL, compile = TRUE)
pred <- modell %>% predict(list(DNAs, Proteins))
classes <- pred>0.5 

modell %>% evaluate(list(DNAs, Proteins), Ys)
Sensitivity(y_true = as.vector(Ys),y_pred =as.vector(classes)%>% as.numeric() ,  positive = "1" )
Specificity(y_true = as.vector(Ys),y_pred =as.vector(classes)%>% as.numeric() ,  positive = "1" )
F1_Score(y_true = as.vector(Ys),y_pred =as.vector(classes)%>% as.numeric() ,  positive = "1" )
AUC(y_pred = pred, y_true = as.vector(Ys))
ConfusionMatrix(y_true = as.vector(Ys),y_pred =as.vector(classes)%>% as.numeric()  )
