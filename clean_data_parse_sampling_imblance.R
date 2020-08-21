library(tidyverse)
library(readxl)

args <- commandArgs(trailingOnly = T)

data <- read_xls(args[1])
dna_tbl  <- read.csv(args[2])
protein_tbl  <- read.csv(args[3])
val_size <- 5

max_dna_length <- 300
max_pro_length <- 2000
random_id <- sample(1:9999,1)
random_id <- 3000
set.seed( random_id )

trans_from_tbl <- function(x, tbl){
  tbl[x == tbl[,2],3]
}

filled_length <- function(x, max_len, tbl){
  c(x %>% strsplit(.,split = "") %>% unlist(),rep("@",max_len-nchar(x))) %>% as.factor() %>% sapply(.,trans_from_tbl,tbl = tbl)
}

data <- data %>% select(`Class Label`, `DNA/RNA`, Protein)

clean_data <- data %>% 
              filter( !{ Protein  %>% grepl("X",.)} ) %>% 
              filter( !{`DNA/RNA` %>% grepl("-",.)} ) %>% 
              filter( !{`DNA/RNA` %>% grepl("B",.)} ) %>% 
              filter( !{`DNA/RNA` %>% grepl("N",.)} ) %>% 
              mutate( `DNA/RNA`  = `DNA/RNA` %>% gsub("U",replacement = "T",.))
  


DNA_array <- clean_data$`DNA/RNA` %>% sapply(.,filled_length, max_len = max_dna_length, tbl = dna_tbl ) %>% t()
Protein_array <- clean_data$Protein %>% sapply(.,filled_length, max_len = max_pro_length, tbl = protein_tbl ) %>% t()
y_data <- clean_data$`Class Label` %>% sapply(.,function(x){if(x == "negative"){0} else {1} }) %>% as.integer()


y_data %>% table

which_1 <-  which(y_data==1) # less
which_0_all <- which(y_data==0) 
which_0_sample_num <- sample( 1:length(which_0_all), length(which_1), replace = F)
which_0 <- which_0_all[which_0_sample_num]
which_0_unselect <- which_0_all[-which_0_sample_num]


sample_num <- sample(1:length(which_1) , length(which_1)%/%val_size, replace = F) 

which_1_test  <- which_1[sample_num ]
which_1_train <- which_1[-sample_num ]
which_0_test  <- which_0[sample_num ]
which_0_train <- c( which_0[-sample_num ],which_0_unselect )

which_1_test_nerver_seen <- which_1_test[1:50]
which_0_test_nerver_seen <-which_0_test[1:50]

which_1_test <- which_1_test[-1:-50]
which_0_test <- which_0_test[-1:-50]

# which_1_test %>% length
# which_1_train %>% length
# which_0_test %>% length
# which_0_train %>% length
# 
# which_1_test_nerver_seen %>% length
# which_0_test_nerver_seen %>% length

y_test = y_data[c( which_0_test,which_1_test)]
y_train = y_data[c( which_0_train,which_1_train)]
y_never_seen = y_data[c( which_1_test_nerver_seen,which_0_test_nerver_seen)]

DNA_test = DNA_array[c( which_0_test,which_1_test),]
DNA_train = DNA_array[c( which_0_train,which_1_train),]
DNA_never_seen = DNA_array[c( which_1_test_nerver_seen,which_0_test_nerver_seen),]

Protein_test = Protein_array[c( which_0_test,which_1_test),]
Protein_train = Protein_array[c( which_0_train,which_1_train),]
Protein_never_seen = Protein_array[c( which_1_test_nerver_seen,which_0_test_nerver_seen),]




#write.csv(tibble( protein_type, protein_code), "protein_code_table.csv")
#write.csv(tibble( DNA_type, DNA_code), "DNA_code_table.csv")
write.csv(DNA_test,"DNA_test.csv" )
write.csv(DNA_train,"DNA_train.csv" )
write.csv(Protein_train,"Protein_train.csv" )
write.csv(Protein_test,"Protein_test.csv" )
write.csv(y_test, "y_test.csv")
write.csv(y_train, "y_train.csv")
write.csv(y_never_seen,"y_never_seen.csv")
write.csv(DNA_never_seen, "DNA_never_seen.csv")
write.csv(Protein_never_seen, "Protein_never_seen.csv")
write_delim( data.frame(random_id), "random_Id.txt")



library(tidyverse)
diversity_plot <- function (sampled_file, Title ){
  tmp <- sampled_file %>% as_tibble() %>% sapply(.,table) %>% lapply(.,function(x){x/nrow(sampled_file)})
  res <- tibble()
  for ( uu in 1:length(tmp) ){
    res <- bind_rows(res,  tmp[[uu]] %>% as.matrix() %>% t()%>% as_tibble() )
  }
  ID <- colnames(res)
  tmpp <- t(res) %>% as_tibble() %>% mutate(ID = ID) %>% gather(., key = position , value, -ID) %>%  replace_na( ., replace = list(value=0) ) %>% filter(  ID != "1" ) %>% filter(value != 0 ) %>% mutate( position = as.numeric(sub("V","",position)) )
  
  p <- ggplot(tmpp, aes(x=position, y=value)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
    geom_bar(aes(x=position, y=value, fill=ID), stat="identity", alpha=0.5)+
    # ylim(-100,120) +
    
    # Custom the theme: no axis title and no cartesian grid
    theme_minimal() +
    theme(
      axis.text = element_text(color="black", size = 34),
      plot.title = element_text(color="black", size=45, face="bold.italic"),
      plot.subtitle = element_text(color="black", size=34),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      legend.position = "none"# This remove unnecessary margin around plot
    ) + ggtitle(Title, subtitle = paste("size",nrow(sampled_file), sep = " = "))
  p
  
  
}
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


a <- diversity_plot(Protein_train, "Protein_train")
b <- diversity_plot(Protein_test, "Protein_test")
#c <- diversity_plot(Protein_never_seen, "Protein_never_seen")
d <- diversity_plot(DNA_train, "DNA_train")
e <- diversity_plot(DNA_test, "DNA_test")
#f <- diversity_plot(DNA_never_seen, "DNA_never_seen")
#ggsave(multiplot(a,b,c,d,e,f,   cols=1), file=paste(random_id,"_sampling_diversity.png", sep = "" ), width=20, height=50,limitsize = FALSE)
ggsave(multiplot(a,b,d,e, cols=1), file=paste(random_id,"_sampling_diversity.png", sep = "" ), width=20, height=50,limitsize = FALSE)

cat( "train = " , length(which_1_train)*2  ,"  test = " , length(which_0_test)*2  )









