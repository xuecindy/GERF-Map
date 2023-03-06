package sound_wave;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class ThreeD_Data {

	static String[] valid_file_format= {"byFA", "byT"};
	
	String headers="";
	int smooth_level=-1;
	int num_T=-1;
	int num_A;
	int num_F;
	float[] Amplitude;
	float[] Frequency;
	float[][][] FireRate; // [num_time][num_F][num_A];

	public ThreeD_Data(String file_address, String format){		
		if(format.equals("byFA")) {
			this.format_byFA(file_address);
		}else if(format.equals("byT")) {
			this.format_byT(file_address);
		}else {
			System.out.println("Format "+ format+ " is not supported");
			System.exit(0);
		}
		
	}
	
	/*
	 * Substract the FT values from "ThreeD_Data before".
	 */
	public void subtract(ThreeD_Data before) {
		// checking dimensions and A/F values
		if(this.num_A!=before.num_A || this.num_T!=before.num_T || this.num_F!=before.num_F ||
				this.smooth_level!=before.smooth_level) {
			System.out.println("Dimensions not match! Can't run ThreeD_Data.subtract(before)");
			System.exit(0);
		}for(int k=0;k<this.num_A;k++) {
			if(Float.compare(this.Amplitude[k], before.Amplitude[k])!=0) {
				System.out.println("Amplitude["+k+"] Not equal"+ this.Amplitude[k]+":"+before.Amplitude[k]);
				System.exit(0);
			}
		}for(int k=0;k<this.num_F;k++) {
			if(Float.compare(this.Frequency[k], before.Frequency[k])!=0) {
				System.out.println("Frequency["+k+"] Not equal"+ this.Frequency[k]+":"+before.Frequency[k]);
				System.exit(0);
			}
		}// Subtracting values
		for(int t_index=0;t_index<this.num_T;t_index++) {
			for(int F_index=0;F_index<this.num_F;F_index++) {
				for(int A_index=0;A_index<this.num_A;A_index++) {
					this.FireRate[t_index][F_index][A_index] -= before.FireRate[t_index][F_index][A_index];
				}
			}
		}
		
	}

	/*
	 * the T-format is the format printed by ThreeD_Data, each block is for a timepoint. 
	 */
	public void format_byT(String file_address) {
		try {
			BufferedReader br=new BufferedReader(new FileReader(file_address));
			String line=br.readLine();
			// retain headers
			while(line.startsWith("#")) {
				this.headers=this.headers+line+"\n";
				line=br.readLine();		
			}
			line=br.readLine(); // skip the line "Time"
			this.num_T=Integer.parseInt(line);
			line=br.readLine(); 
			line=br.readLine(); // skip the line "Amplitude"
			String[] temp=line.split(" ");
			this.num_A=temp.length;
			this.Amplitude=new float[this.num_A];
			System.out.println("num_A="+this.num_A);
			for(int k=0;k<this.num_A;k++) {
				this.Amplitude[k]=Float.parseFloat(temp[k]);
			}
			line=br.readLine(); 
			line=br.readLine(); // skip the line "Frequency"
			temp=line.split(" ");
			this.num_F=temp.length;
			this.Frequency=new float[this.num_F];
			System.out.println("num_F="+this.num_F);
			for(int k=0;k<this.num_F;k++) {
				this.Frequency[k]=Float.parseFloat(temp[k]);
			}
			this.FireRate=new float[this.num_T][this.num_F][this.num_A];
			for(int t_index=0;t_index<this.num_T;t_index++) {
				line=br.readLine(); // the line of "T=?"
				//System.out.println(line); // print T=0 line
				String[] t_line=line.split("=");
				if(!(t_line[0].equals("T") && t_line[1].equals(""+t_index))){
					System.out.println("Block header "+line+" is wrong! "+ t_index);
					System.exit(0);
				}for(int a_index=0;a_index<this.num_A;a_index++) {
					//System.out.println(a_index+", "+t_index+", "+line);
					String[] f_line=br.readLine().split(" ");
					if(f_line.length!=this.num_F) {
						System.out.println("f_line.length!=this.num_F");
						System.exit(0);
					}for(int f_index=0;f_index<this.num_F;f_index++) {
						this.FireRate[t_index][f_index][a_index]=Float.parseFloat(f_line[f_index]);
					}
				}
			}
			System.out.println("Read file byT from "+file_address);
		}catch (IOException e) {  e.printStackTrace();}
	}
	
	/*
	 * the FA-format is the raw txt file directed translated from binary file.
	 * each line is a record of data with fixed A and F at different timepoint
	 */
	public void format_byFA(String file_address){
		HashSet<String> Freqs = new HashSet<String>();
		HashSet<String> Amps = new HashSet<String>();
		HashMap<String, String[]> data= new HashMap<String, String[]>();
		try {
			BufferedReader br=new BufferedReader(new FileReader(file_address));
			String line=br.readLine();
			// retain headers
			while(line.startsWith("#")) {
				this.headers=this.headers+line+"\n";
				line=br.readLine();		
			}
			//create the data hashmap
			while(line!=null) {
				String[] AF_line=line.split(" ");
				AF_line[1]=rounding_freq(Float.parseFloat(AF_line[1]))+"";
				AF_line[3]=rounding(Float.parseFloat(AF_line[3]))+"";
				Freqs.add(AF_line[1]);
				Amps.add(AF_line[3]);
				String[] line_data=br.readLine().split(" ");
				if(this.num_T==-1) {
					this.num_T=line_data.length;
				}else {
					if(this.num_T!=line_data.length) {
						System.out.println("WRONG! this.num_time!=line_data.length");
					}
				}
				data.put(AF_line[1]+"_"+AF_line[3], line_data);
				line=br.readLine();
			}

		} catch (IOException e) {  e.printStackTrace();}
		
		this.num_A=Amps.size();
		this.num_F=Freqs.size();
		this.Amplitude=new float[this.num_A];
		this.Frequency=new float[this.num_F];
		int a_index=0;
		for(String key: Amps) {
			this.Amplitude[a_index++]=Float.parseFloat(key);
		}
		Arrays.sort(this.Amplitude);
		int f_index=0;
		for(String key: Freqs) {
			this.Frequency[f_index++]=Float.parseFloat(key);
		}
		Arrays.sort(this.Frequency);

		this.FireRate=new float[this.num_T][this.num_F][this.num_A];

		for(int am_index=0;am_index<this.num_A;am_index++) {
			for(int fr_index=0;fr_index<this.num_F;fr_index++) {
				// take the data from this pair of A and F
				String[] temp=data.get(rounding_freq(this.Frequency[fr_index])+"_"+rounding(this.Amplitude[am_index]));
				for(int time_index=0;time_index<this.num_T;time_index++) {
					//this.FireRate[time_index][fr_index][am_index]=Float.parseFloat(temp[time_index]);
					if(temp!=null) {
						this.FireRate[time_index][fr_index][am_index]=Float.parseFloat(temp[time_index]);
					}else {
						this.FireRate[time_index][fr_index][am_index]=0;
					}
				}
			}
		}
		System.out.println("Read file byFA from "+file_address);
	}
	
	
	public static void rounding(float[] input) {
		for(int i=0;i<input.length;i++) {
			DecimalFormat df = new DecimalFormat("#.##");
			input[i]=Float.valueOf(df.format(input[i]));
		}
	}

	public static float rounding(float input) {
		DecimalFormat df = new DecimalFormat("#.##");
		return Float.valueOf(df.format(input));
	}
	public static float rounding_freq(float input) {
		DecimalFormat df = new DecimalFormat("#.####");
		return Float.valueOf(df.format(input));
	}

	/*
	 * the method that smoothes the 3D matrix 
	 */
	public void smoothing(int level) {
		if(level==0) {
			return;
		}
		this.smooth_level=level;
		float[][][] new_data= new float[this.num_T][this.num_F][this.num_A];
		for(int time=0;time<this.num_T;time++) {
			for(int frequency=0;frequency<this.num_F;frequency++) {
				for(int amplitude=0;amplitude<this.num_A;amplitude++) {
					int[] t=legal_indexes(level, time, this.num_T);
					int[] f=legal_indexes(level, frequency, this.num_F);
					int[] a=legal_indexes(level, amplitude, this.num_A);
					float average=0;
					for(int i : t) {
						for(int j : f) {
							for(int k : a) {
								average=average+this.FireRate[i][j][k];
							}
						}
					}
					average=average/(t.length*f.length*a.length);
					average=rounding_freq(average);
					new_data[time][frequency][amplitude]=average;
				}
			}
		}
		this.FireRate=new_data;
	}

	/*
	 * Figure out the legal indexes for smoothing 
	 */
	public static int[] legal_indexes(int level, int index, int total_length) {
		int min=Math.max(index-level, 0);
		int max=Math.min(index+level, total_length-1);
		int[] output=new int[max-min+1];
		for(int i=0;i<output.length;i++) {
			output[i]=min+i;
		}
		return output;
	}
	
	/*
	 * 	output to .txt file in byT format
	 */
	public void write2file(String file_address) {
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(file_address));
			bw.write(this.headers);
			if(this.smooth_level!=-1) bw.write("# Level = " +this.smooth_level+"\n");
			// write in the Time, Amplitude and Frequency below the headers
			bw.write("Time\n"+this.num_T+"\n");
			bw.write("Amplitude: "+this.num_A+"\n");
			for(int k=0;k<this.num_A;k++) {
				bw.write(this.Amplitude[k]+" ");
			}bw.write("\nFrequency: "+this.num_F+"\n");
			for(int k=0;k<this.num_F;k++) {
				bw.write(this.Frequency[k]+" ");
			}bw.write("\n");
			for(int time_index=0;time_index<this.num_T;time_index++) {
				bw.write("T="+time_index+"\n");
				for(int a_index=0;a_index<this.num_A;a_index++) {
					for(int f_index=0;f_index<this.num_F;f_index++) {
						bw.write(FireRate[time_index][f_index][a_index]+" ");
					}
					bw.write("\n");
				}
			}
			bw.flush();
			bw.close();

		}catch(Exception e) {
			e.printStackTrace();
		}
		System.out.println("Output file to "+file_address);
	}
	
	/*
	 * 	output to csv format
	 */
	public void filewritecsv(String output) {
		try{
			FileWriter fw = new FileWriter(output);
			BufferedWriter bw = new BufferedWriter(fw);
			//			Header
			//			Array
			for(int time_index=0;time_index<this.num_T;time_index++) {
//				bw.write("[T="+(time_index+1)+"ms]\n");
				bw.write("\n");
				
				for(int f_index=0;f_index<this.num_F;f_index++) {
					if(f_index!=0) {
						bw.write(",");
					}
					bw.write(Frequency[f_index]+"");
				}
				bw.write("\n");
				for(int a_index=0;a_index<this.num_A;a_index++) {
					bw.write(Amplitude[a_index]+"");
					for(int f_index=0;f_index<this.num_F;f_index++) {
						bw.write(","+FireRate[time_index][f_index][a_index]+"");

					}
					bw.write("\n");
				}
			}
			bw.flush();
			bw.close();

		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public void filewritecsv_flip(String output) {
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(output));
			//			Header
			//			Array
			for(int time_index=0;time_index<this.num_T;time_index++) {
//				bw.write("[T="+(time_index+1)+"ms]\n");
				bw.write("\n");
				
				for(int f_index=0;f_index<this.num_F;f_index++) {
					if(f_index!=0) {
						bw.write(",");
					}
					bw.write(Frequency[f_index]+"");
				}
				bw.write("\n");
				for(int a_index=this.num_A-1;a_index>=0;a_index--) {
					bw.write(Amplitude[a_index]+"");
					for(int f_index=0;f_index<this.num_F;f_index++) {
						bw.write(","+FireRate[time_index][f_index][a_index]+"");

					}
					bw.write("\n");
				}
			}
			bw.flush();
			bw.close();

		}catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {

		String raw_txt= "/Users/fan/Dropbox/Zhou/ScienceFair/G_RA16-1_001Control25Jo1.raw.txt";
		String data_3d= "/Users/fan/Dropbox/Zhou/ScienceFair/G_RA16-1_001Control25Jo1.txt";
		String data_smoothed= "/Users/fan/Dropbox/Zhou/ScienceFair/G_RA16-1_001Control25Jo1.smoothed.txt";
		String data_csv= "/Users/fan/Dropbox/Zhou/ScienceFair/G_RA16-1_001Control25Jo1.csv.txt";
		String data_flip_csv= "/Users/fan/Dropbox/Zhou/ScienceFair/G_RA16-1_001Control25Jo1_flip.csv.txt";
		int level=1;
		ThreeD_Data raw_data=new ThreeD_Data(raw_txt, "byFA");
//		raw_data.out2filewriting(data_3d, -1);
		//		ThreeD_Data formatted_data=new ThreeD_Data(data_3d, true);
		raw_data.smoothing(level);
//		raw_data.out2filewriting(data_smoothed, level);
		raw_data.filewritecsv(data_csv);
		raw_data.filewritecsv_flip(data_flip_csv);
		//		formatted_data.out2filewriting(data_smoothed);

	}

}

