library(plotly);

root_folder="/Users/fan/Dropbox/Zhou/ScienceFair/Data2023-02-17/";
setwd(root_folder);
IDs= c("16","18","19","20","21","22","23","25","26","28","29","31","32","36","37","38","39","40","43");
#prefix="Control";
prefix="Post";
names <- c("freq_BF", "freq_BW", "time_latency","time_50duration","time_rising_slope10",
           "amp_threshold", "amp_DR_x", "amp_DR_slope");
for(i in 1:length(names)){
  for(k in 1:length(IDs)){
    data <- read.csv(file=paste("sample_", IDs[k],"/", prefix, IDs[k],
                                "_contour_files/",names[i],".csv", sep=""), header=FALSE);
    pdf(file=paste("sample_", IDs[k],"/", prefix, 
                   IDs[k],"_contour_files/",prefix,IDs[k],"_",names[i],".heatmap.pdf",sep =""));
    heatmap(as.matrix(data),Colv = NA, Rowv = NA, main=paste(prefix,IDs[k],names[i],sep=""));
    dev.off();
  }
}

