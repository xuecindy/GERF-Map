library(plotly);

root_folder="/Users/fan/Dropbox/Zhou/ScienceFair/Data2023-02-17/";
setwd(root_folder);
IDs= c("16","18","19","20","22","23","25","26","28","29","31","32","36","37","38","39","43")
IDs= c("16","18","19","20","22","23","25","26","28","29","31","32","36","37","38","39","43");
prefix="Control";
#prefix="Post";
#prefix="Diff";
names <- c("freq_BF", "freq_BW", "freq_noise", 
           "time_latency","time_50duration","time_rising_slope10", "time_noise", 
           "amp_threshold", "amp_DR_x", "amp_DR_slope", "amp_noise");
names <- c("time_50duration","time_rising_slope10", "time_noise",
           "amp_threshold", "amp_noise");
names <- c("amp_threshold" );
for(i in 1:length(names)){
  for(k in 1:length(IDs)){
    data <- read.csv(file=paste("sample_", IDs[k],"/", prefix, IDs[k],
                    "_contour_files/",names[i],".csv", sep=""), header=FALSE);
#    fig <- plot_ly(z = as.matrix(data), contours = list(showlabels = TRUE),line = list(smoothing = 1.5), type = "contour");
    fig <- plot_ly(z = as.matrix(data), contours = list(showlabels = TRUE), type = "contour");
    fig <- fig %>% layout(title = paste(prefix,IDs[k],names[i],sep = ""));
    plotly::export(p = fig, file = paste("sample_", IDs[k],"/", prefix, 
                    IDs[k],"_contour_files/",prefix,IDs[k],"_",names[i],".png",sep =""));
  }
}

