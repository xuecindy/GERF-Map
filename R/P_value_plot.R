root_folder="/Users/fan/Dropbox/Zhou/ScienceFair/Data2023-02-17/";
setwd(root_folder);

data=read.csv("Combined/t_test.comb.csv", header=FALSE, skip=8);
time=c(1:100);
amps=2*c(7:42)

pdf("Combined/t.pdf", width=5, height=10);
par(mfrow= c(5,1))
plot(x=amps, y=-log10(data[1,1:36]), col = 4, pch=2, lwd = 3, xlab="Amplitude (dB)",ylab="-log10(P-value)", main="Time: 50 Percent Duration")
abline(a=-log10(0.05/length(amps)/11), b=0, col=2)
plot(x=amps, y=-log10(data[2,1:36]), col = 4, pch=2, lwd = 3, xlab="Amplitude (dB)",ylab="-log10(P-value)", main="Time: Rising Slope")
abline(a=-log10(0.05/length(amps)/11), b=0, col=2)
plot(x=amps, y=-log10(data[3,1:36]), col = 4,  pch=2, lwd = 3, xlab="Amplitude (dB)",ylab="-log10(P-value)",main="Time: Noise/Signal")
abline(a=-log10(0.05/length(amps)/11),  b=0, col=2)
plot(x=time, y=-log10(data[4,]), col = 4, pch=1, lwd = 3, xlab="Time (ms)", ylab="-log10(P-value)", main="Amp: Threshold")
abline(a=-log10(0.05/length(time)/11),  b=0, col=2)
plot(x=time, y=-log10(data[5,]), col = 4, pch=1, lwd = 3, xlab="Time (ms)",ylab="-log10(P-value)", main="Amplitude: Noise/Signal")
abline(a=-log10(0.05/length(time)/11), b=0, col=2)
dev.off()

pdf("Combined/ca.pdf", width=5, height=10);
par(mfrow= c(5,1))
data=read.csv("Combined/ca_trend.comb.csv", header=FALSE, skip=8);
plot(x=amps, y=-log10(data[1,1:36]), col = 4, pch=2, lwd = 3, xlab="Amplitude (dB)",ylab="-log10(P-value)", main="Time: 50 Percent Duration")
abline(a=-log10(0.05/length(amps)/11), b=0, col=2)
plot(x=amps, y=-log10(data[2,1:36]), col = 4, pch=2, lwd = 3, xlab="Amplitude (dB)",ylab="-log10(P-value)", main="Time: Rising Slope")
abline(a=-log10(0.05/length(amps)/11), b=0, col=2)
plot(x=amps, y=-log10(data[3,1:36]), col = 4,  pch=2, lwd = 3, xlab="Amplitude (dB)",ylab="-log10(P-value)",main="Time: Noise/Signal")
abline(a=-log10(0.05/length(amps)/11),  b=0, col=2)
plot(x=time, y=-log10(data[4,]), col = 4, pch=1, lwd = 3, xlab="Time (ms)", ylab="-log10(P-value)", main="Amp: Threshold")
abline(a=-log10(0.05/length(time)/11),  b=0, col=2)
plot(x=time, y=-log10(data[5,]), col = 4, pch=1, lwd = 3, xlab="Time (ms)",ylab="-log10(P-value)", main="Amplitude: Noise/Signal")
abline(a=-log10(0.05/length(time)/11), b=0, col=2)
dev.off()