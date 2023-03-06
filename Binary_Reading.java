package sound_wave;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;

public class Binary_Reading {

	private static final int BUFFER_SIZE = 4096; // 4KB
	
	public static int fromByteArray(byte[] bytes) {
	     return bytes[0] << 24 | (bytes[1] & 0xFF) << 16 | (bytes[2] & 0xFF) << 8 | (bytes[3] & 0xFF);
	}
	
	public static int integer(byte[] b) {
		int l = (int)b[0] & 0xFF;
		l += ((int)b[1] & 0xFF) << 8;
		l += ((int)b[2] & 0xFF) << 16;
		l += ((int)b[3] & 0xFF) << 24;
		return l;
	}
	
	public static int smallint(byte[] b) {
		int out=b[1]*256+b[0];
		return out;
	}
	
	public static String text(byte[] b) {
		String s = new String(b);
		return s;
	}
	
	
	public static void binary2txt(String input_file, String output_file) {
		try {
			 FileWriter fw = new FileWriter(output_file);
			 BufferedWriter bw = new BufferedWriter(fw);
			 InputStream inputStream = new FileInputStream(input_file);

			 byte[] data=inputStream.readAllBytes();
			 byte[] length= {data[0],data[1],data[2],data[3]};
			 byte[] two= {data[4],data[5]};
			 byte[] three= {data[6],data[7],data[8]};
			 byte[] six= {data[9],data[10], data[11], data[12], data[13], data[14]};
			 
			 int file_length=integer(length);
			 int head_length=smallint(two);
			 
			 two[0]=data[18]; two[1]=data[19];
			 int data_record_length=smallint(two);
			 
			 two[0]=data[20]; two[1]=data[21];
			 int one_record_length=smallint(two);
			
			 two[0]=data[22]; two[1]=data[23];
			 int datasets=smallint(two);
	
			 two[0]=data[24]; two[1]=data[25];
			 int starting_point=smallint(two);
			 
			 bw.write("## File Length = "+file_length+"\n");
			 bw.write("## Head Length = "+head_length+"\n");
			 bw.write("## Frequency Unit = "+text(three)+"\n");
			 bw.write("## Amplitude Unit = "+text(six)+"\n");
			 bw.write("## Data-starting point in byte = "+starting_point+"\n");
			 three[0]=data[15]; three[1]=data[16]; three[2]= data[17];
			 bw.write("## Data type = "+text(three)+"\n");
			 bw.write("# Data record length = "+data_record_length+"\n");
			 bw.write("# Length of one record = "+one_record_length+"\n");
			 bw.write("# Number of data set = "+datasets+"\n");
			 
			 
			int safe_buffer=500;
			float[][] output=new float[datasets+safe_buffer][data_record_length+2];
			 
			 for(int i=head_length;i<data.length-3;i=i+4) {				 
				 int set=Math.max((i-head_length)/(4*data_record_length+8), 0);
				 int index=Math.max(((i-head_length)/4)%(data_record_length+2), 0);				 
				 byte[] temp= {data[i],data[i+1],data[i+2],data[i+3]};				
				 output[set][index]=ByteBuffer.wrap(temp).order(ByteOrder.LITTLE_ENDIAN).getFloat();
			 }			 
			 for(int i=0;i<datasets;i++) {
				 bw.write("Frequency: "+ output[i][0]);
				 bw.write(" Amplitude: "+ output[i][1]+"\n");
				 for(int j=2;j<output[0].length;j++) {
					 bw.write(output[i][j]+" ");
				 }bw.write("\n");
				 bw.flush();
			 }bw.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		try {
			String folder="/Users/fan/Dropbox/Zhou/ScienceFair/Data2023-02-17/";
			File[] qua=new File(folder+"qua/").listFiles();
			for(int k=0;k<qua.length;k++) {
				String binary_file=qua[k].toString();
				String name=binary_file.split("/")[binary_file.split("/").length-1];
				String raw_txt = folder+"txt/"+name+".txt";
				binary2txt(binary_file, raw_txt);
			}
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		
//		String data_smoothed= "/Users/fan/Dropbox/Zhou/ScienceFair/G_RA16-1_001Control25Jo1.smoothed.txt";
		
	}
	
}

