package sound_wave;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;

//import org.apache.commons.math4.legacy.stat.inference.*;
//import org.apache.commons.math4.legacy.distribution.MultivariateNormalDistribution;
//import org.apache.commons.statistics.distribution.NormalDistribution;
//import org.apache.commons.statistics.distribution.*;
//import org.apache.commons.rng.sampling.distribution.*;
//import org.apache.commons.numbers.*;

public class StatTests {

	// removed poor quality sample 21 and 40 (Comparing to Analysis.IDs)
	static final String[] IDs={"16","18","19","20","22","23",
					"25","26","28", "29","31","32",
					"36","37","38","39","43"};
	static final double[] exposure_freq= {15.1, 17.4, 13.1, 17.4, 8.7, 11.4,
					10, 13.1, 15.1, 10, 13.1, 13.1,
					17.4, 10, 11.4, 11.4, 11.4};
	static final String[] names= {"freq_BF", "freq_BW", "freq_noise", 
			"time_latency","time_50duration","time_rising_slope10", "time_noise",
			"amp_threshold", "amp_DR_x", "amp_DR_slope", "amp_noise"};
	static final String root_folder=Analysis.root_folder;
	
	static final double[] freqs= {2.5, 2.8717, 3.2988, 3.7893, 4.3528, 5.0, 5.7435, 6.5975, 7.5786, 8.7055,
			10.0, 11.487, 13.1951, 15.1572, 17.411, 20.0, 22.974, 26.3902, 30.3143, 34.822, 40.0};
	static final String[] amps= {"14.0", "16.0", "18.0", "20.0", "22.0", "24.0", "26.0", "28.0", "30.0", "32.0", 
			"34.0", "36.0", "38.0", "40.0", "42.0", "44.0", "46.0", "48.0", "50.0", "52.0",	"54.0", "56.0", "58.0", 
			"60.0", "62.0", "64.0", "66.0", "68.0", "70.0", "72.0", "74.0", "76.0", "78.0", "80.0", "82.0", "84.0" };
	static final int time_num=100;
	static final double base_weight=5;
		
	/*
	 * Based on the exposure frequency (static double[] exposure_freq), set up 
	 * the weight for C-A test for trend.
	 * 
	 * The matching frequency will be set with the highest weight, and the others'
	 * weight will be decided based on their distance to the exposure frequency.  
	 */
	public static double[] set_weights(int ID_index) {
		double[] weight=StatTests.freqs.clone();
		for(int i=0;i<weight.length;i++) {
			if(weight[i]>exposure_freq[ID_index]) {
				double diff=weight[i]-exposure_freq[ID_index];  // distance from exposure freq
				weight[i] = exposure_freq[ID_index]-diff;  // lower weight if further away
			}
		}
		double minimal=weight[weight.length-1];
		for(int i=0;i<weight.length;i++) {
			if(minimal > weight[i])minimal = weight[i];
		}
		for(int i=0;i<weight.length;i++) {
			weight[i] = weight[i] + base_weight-minimal;
		}
		return weight;
	}
	/*
	 * returns p-value of a paired two-sample t-test.
	 * Tested the same as R function t.test(pair=TRUE, alternative="two.sided") 
	 */
	public static double t_test(double[] sample1, double[] sample2) {
		if(sample1.length!=sample2.length)return Double.NaN;
		TTest tester= new TTest();
		//double t=tester.pairedT(sample1, sample2);
		return tester.pairedTTest(sample1, sample2);
	}
	
	/*
	 * CochranArmitageTest(alternative="two.sided"))
	 * based on formula at https://en.wikipedia.org/wiki/Cochran–Armitage_test_for_trend
	 * tested to be the same as R function DescTools::CochranArmitageTest()
	 */
	public static double ca_trend(double[] sample1, double[] sample2, double[] weight) {
		if(sample1.length!=sample2.length)return Double.NaN;
		double r1=0, r2=0; // row sum
		double[] c= new double[sample1.length]; //column sum
		for(int k=0;k<c.length;k++) {
			c[k]=sample1[k]+sample2[k];
			r1 = r1+sample1[k];
			r2 = r2+sample2[k];
		}
		double N = r1+r2;
		double T=0; // The trend test statistic 
		double var_T=0; // The variance of T = (r1*r2/N)(term1 - 2*term2)
		double term1=0, term2=0;
		for(int i=0;i<c.length;i++) {
			T = T+weight[i]*(sample1[i]*r2 - sample2[i]*r1);
			term1= term1+weight[i]*weight[i]*c[i]*(N-c[i]);
		}
		for(int i=0;i<c.length-1;i++) {
			for(int j=i+1;j<c.length;j++) {
				term2 = term2+weight[i]*weight[j]*c[i]*c[j];
			}
		}
		var_T = (r1*r2/N)*(term1 -2*term2);
		// T/sqrt(var_T)~ N(0,1)
		NormalDistribution nd = new NormalDistribution(0, 1);
		double Z = T/Math.sqrt(var_T);
		double cdf= nd.cumulativeProbability(Z); 
		//two-sided in N(0,1)
		double p_value=2*((cdf<0.5)?(cdf):(1-cdf));
		return p_value;
	}
	
	/*
	 * multiple samples are combined using Fisher's combined probability test 
	 * 	Based on the formula in https://en.wikipedia.org/wiki/Fisher%27s_method
	 * 	tested to be the same as the R function poolr::fisher
	 */
	
	public static double Fisher_combined(double[] p_tobe_combined) {
		double stat=0;
		for(int k=0;k<p_tobe_combined.length;k++) {
			if(p_tobe_combined[k]==0)return 0;  // return 0 is any of the original is 0.
			stat=stat-2*Math.log(p_tobe_combined[k]);
		}ChiSquaredDistribution chisq = new ChiSquaredDistribution(2*p_tobe_combined.length);
		double combined_p=1-chisq.cumulativeProbability(stat);
		if(combined_p>0.5)combined_p=1-combined_p;
		// ad hoc code to ensure the plotting function works
		if(Double.compare(combined_p,0) == 0) 
			return (1E-15)*Math.pow(100, Math.random());
		return combined_p;
	}
	
	/*
	 * Load a CSV file into an array. 
	 * Check the number of columns in each row to be the same if "check_length == TRUE"
	 */
	public static double[][] load_csv(String csv_file, boolean check_length) {
		double[][] data = null; 
		ArrayList<double[]> buffer= new ArrayList<double[]>();
		try {
			BufferedReader br= new BufferedReader(new FileReader(csv_file));
			String line=br.readLine();
			while(line.startsWith("#"))line=br.readLine(); // skip headers
			while(line!=null) {
				String[] tmp=line.split(",");
				double[] data_line=new double[tmp.length];
				for(int k=0;k<tmp.length;k++)data_line[k]=Double.parseDouble(tmp[k]);
				buffer.add(data_line);
				line=br.readLine();
			}br.close();
		}catch(Exception e) {e.printStackTrace();}
		data=new double[buffer.size()][];
		for(int i=0;i<buffer.size();i++)data[i]=buffer.get(i);
		int row_length=data[0].length;
		if(check_length) {
			for(int i=1;i<data.length;i++) {
				if(row_length!=data[i].length)
					System.out.println("Warning: load_csv():"+csv_file+": row"+i+":"+data[i].length);
			}
		}
		return data;
	}
	
	/*
	 * Quantifying the test significant level based on the intuitive effects 
	 * 
	 * First we test within a sample along Frequency axis (i.e., 21 samples) for Time and Amplitude 
	 * domains. Second, the outcome of these within-sample tests are combined by Fisher's Method.  
	 * 
	 * Tests within a single sample. 
	 * 		Both paired two sample t-test and Cochran-Armitage Test for Trend are implemented.
	 * 		(In trend test, the frequency used for 1-hour exposure will be set to be the highest 
	 * 		weight, and others will be set accordingly).
	 * 		Here the following categories will be tested:
	 * 			(1) Time domain: 50-duration; rising slope; noise/signal
	 * 			(2) Amp domain: threshold; noise/signal
	 * 
	 * There will be two output files in each sample_ID folder: 
	 * 		TestID_pvaule_files/t_test.csv
	 * 		TestID_pvaule_files/ca_trend.csv
	 * Additionally, Fisher's combined results will be stored in 
	 * 		root_folder/Combined/t_test.combined.csv
	 * 		root_folder/Combined/ca_trend.combined.csv
	 * 
	 */
	public static void test_along_freq() {
		String pre="Control";
		String post="Post";
		String test="Test";
		String[] time_test_names= {"time_50duration","time_rising_slope10", "time_noise"};
		String[] amp_test_names={"amp_threshold", "amp_noise"};
		double[][] weights = new double[IDs.length][freqs.length];
		for(int ID_index=0;ID_index<IDs.length;ID_index++)
			weights[ID_index]=set_weights(ID_index);
		try {
			// test individual samples
			for(int ID_index=0; ID_index<IDs.length; ID_index++) {
				String working_folder=root_folder+"sample_"+IDs[ID_index]+"/";
				//File working_folder_file=new File(working_folder);
				String pre_contour_folder= working_folder+ pre+IDs[ID_index]+"_contour_files/";
				String post_contour_folder=working_folder+post+IDs[ID_index]+"_contour_files/";
				String test_folder=working_folder+test+IDs[ID_index]+"_pvalue_files/";
				File test_folder_file=new File(test_folder);
				if(!test_folder_file.exists())test_folder_file.mkdir();
				BufferedWriter bw_t= new BufferedWriter(new FileWriter(test_folder+"t_test.csv"));	
				BufferedWriter bw_ca= new BufferedWriter(new FileWriter(test_folder+"ca_trend.csv"));
				// write the headers
				bw_t.write("### Nominal P-values of T-Test for Time domain categories for each Amplitude. "
						+ "Each line is for a category and each column is for an Amplitude value.\n"
						+ "### Nominal P-values of T-Test for Amplitude domain categories for each Time. "
						+ "Each line is for a category and each column is for an Time value.\n");
				bw_ca.write("### Nominal P-values of CA-Trend-Test for time domain categories for each Amplitude. "
						+ "Each line is for a category and each column is for an Amplitude value.\n"
						+ "### Nominal P-values of CA-Trend-Test for Amplitude domain categories for each Time. "
						+ "Each line is for a category and each column is for an Time value.\n");
				for(int time_tn_index=0;time_tn_index<time_test_names.length;time_tn_index++) {
					bw_t.write("# Line "+(time_tn_index+1)+": "+time_test_names[time_tn_index]+"\n");
					bw_ca.write("# Line "+(time_tn_index+1)+": "+time_test_names[time_tn_index]+"\n");
				}for(int amp_tn_index=0;amp_tn_index<amp_test_names.length;amp_tn_index++) {
					bw_t.write("# Line "+(time_test_names.length+amp_tn_index+1)+": "+amp_test_names[amp_tn_index]+"\n");
					bw_ca.write("# Line "+(time_test_names.length+amp_tn_index+1)+": "+amp_test_names[amp_tn_index]+"\n");
				}
				// calculate Time domain categories (for each Amplitude)
				for(int time_tn_index=0;time_tn_index<time_test_names.length;time_tn_index++) {
					double[][] time_tn_pre = load_csv(pre_contour_folder+time_test_names[time_tn_index]+".csv", true);
					double[][] time_tn_post=load_csv(post_contour_folder+time_test_names[time_tn_index]+".csv", true);
					for(int amp_index=0;amp_index<amps.length;amp_index++) {
						// prepare data to carry out t-test and ca_trend test:
						double[] pre_col=new double[freqs.length];
						double[] post_col=new double[freqs.length];
						for(int freq_index=0;freq_index<freqs.length;freq_index++) {
							pre_col[freq_index] = time_tn_pre[freq_index][amp_index];
							post_col[freq_index]=time_tn_post[freq_index][amp_index];
						}// conduct the tests
						double t_pvalue=t_test(pre_col, post_col);
						double ca_pvalue=ca_trend(pre_col, post_col, weights[ID_index]);
						bw_t.write(t_pvalue+((amp_index==(amps.length-1))?"\n":","));
						bw_ca.write(ca_pvalue+((amp_index==(amps.length-1))?"\n":","));
					}System.out.println("Done with "+time_test_names[time_tn_index]);
				}System.out.println("Finished Time Domain for Sample "+IDs[ID_index]+".");
				// calculate Amplitude domain categories (for each Time)
				for(int amp_tn_index=0;amp_tn_index<amp_test_names.length;amp_tn_index++) {
					double[][] amp_tn_pre = load_csv(pre_contour_folder+amp_test_names[amp_tn_index]+".csv", true);
					double[][] amp_tn_post=load_csv(post_contour_folder+amp_test_names[amp_tn_index]+".csv", true);
					for(int time_index=0;time_index<time_num;time_index++) {
						// prepare data to carry out t-test and ca_trend test:
						double[] pre_col=amp_tn_pre[time_index].clone();
						double[] post_col=amp_tn_post[time_index].clone();
						// conduct the tests
						double t_pvalue=t_test(pre_col, post_col);
						double ca_pvalue=ca_trend(pre_col, post_col, weights[ID_index]);
						bw_t.write(t_pvalue+((time_index==(time_num-1))?"\n":","));
						bw_ca.write(ca_pvalue+((time_index==(time_num-1))?"\n":","));
					}System.out.println("Done with "+amp_test_names[amp_tn_index]);
				}System.out.println("Finished Amplitude Domain for Sample "+IDs[ID_index]+".");
				bw_t.close(); bw_ca.close();
			}System.out.println("FINISHED tests within individual samples!\n");
			// combine the above t-test or trend-test p-values across samples using Fisher's Method
			String combined_folder=root_folder+"Combined/";
			File combined_folder_file=new File(combined_folder);
			if(!combined_folder_file.exists())combined_folder_file.mkdir();
			BufferedWriter bw_t_comb= new BufferedWriter(new FileWriter(combined_folder+"t_test.comb.csv"));	
			BufferedWriter bw_ca_comb= new BufferedWriter(new FileWriter(combined_folder+"ca_trend.comb.csv"));	
			// write the headers
			bw_t_comb.write("### Fisher's Method combining samples: ");
			for(int ID_index=0;ID_index<IDs.length;ID_index++) {
				bw_t_comb.write(IDs[ID_index]+((ID_index==IDs.length-1)?"\n":";"));
			}bw_t_comb.write("### Combined P-values of T-Test for Time domain categories for each Amplitude. "
					+ "Each line is for a category and each column is for an Amplitude value.\n"
					+ "### Combined P-values of T-Test for Amplitude domain categories for each Time. "
					+ "Each line is for a category and each column is for an Time value.\n");
			bw_ca_comb.write("### Fisher's Method combining samples: ");
			for(int ID_index=0;ID_index<IDs.length;ID_index++)
				bw_ca_comb.write(IDs[ID_index]+((ID_index==IDs.length-1)?"\n":";"));
			bw_ca_comb.write("### Combined P-values of CA-Trend-Test for time domain categories for each Amplitude. "
					+ "Each line is for a category and each column is for an Amplitude value.\n"
					+ "### Combined P-values of CA-Trend-Test for Amplitude domain categories for each Time. "
					+ "Each line is for a category and each column is for an Time value.\n");
			for(int time_tn_index=0;time_tn_index<time_test_names.length;time_tn_index++) {
				bw_t_comb.write("# Line "+(time_tn_index+1)+": "+time_test_names[time_tn_index]+"\n");
				bw_ca_comb.write("# Line "+(time_tn_index+1)+": "+time_test_names[time_tn_index]+"\n");
			}for(int amp_tn_index=0;amp_tn_index<amp_test_names.length;amp_tn_index++) {
				bw_t_comb.write("# Line "+(time_test_names.length+amp_tn_index+1)+": "+amp_test_names[amp_tn_index]+"\n");
				bw_ca_comb.write("# Line "+(time_test_names.length+amp_tn_index+1)+": "+amp_test_names[amp_tn_index]+"\n");
			}
			// load individual test p-values 
			double[][][] t_test_inds=new double[IDs.length][][];
			double[][][] ca_test_inds=new double[IDs.length][][];
			for(int ID_index=0;ID_index<IDs.length;ID_index++) {
				String test_folder=root_folder+"sample_"+IDs[ID_index]+"/"+test+IDs[ID_index]+"_pvalue_files/";
				t_test_inds[ID_index]=load_csv(test_folder+"t_test.csv", false);
				ca_test_inds[ID_index]=load_csv(test_folder+"ca_trend.csv", false);
			}// combine p-values for each categories:
			int num_categories=t_test_inds[0].length;
			for(int i=0;i<num_categories;i++) {
				int num_values=t_test_inds[0][i].length;
				for(int j=0;j<num_values;j++) { 
					// Fisher's Method for each category, each value
					double[] t_p_values= new double[IDs.length];
					double[] ca_p_values=new double[IDs.length];
					for(int ID_index=0;ID_index<IDs.length;ID_index++) {
						t_p_values[ID_index] = t_test_inds[ID_index][i][j];
						ca_p_values[ID_index]=ca_test_inds[ID_index][i][j];
					}
					bw_t_comb.write(Fisher_combined(remove_NaN(t_p_values))+((j==num_values-1)?"\n":","));
					bw_ca_comb.write(Fisher_combined(remove_NaN(ca_p_values))+((j==num_values-1)?"\n":","));
				}System.out.println("Finished Combining Category: "+ i);
			}System.out.println("FINISHED All Combining!");
			bw_t_comb.close(); bw_ca_comb.close();
		}catch(Exception e) {e.printStackTrace();}
	}
	
	public static double[] remove_NaN(double[] input) {
		ArrayList<Double> valid = new ArrayList<Double>();
		for(int i=0;i<input.length;i++) {
			if(!Double.isNaN(input[i])) valid.add(input[i]);
		}double[] output=new double[valid.size()];
		for(int k=0;k<output.length;k++) {
			output[k]=valid.get(k);
		}return output;
	}
	public static void main(String[] args) {
//		double[] x1= {0.26930718, -0.05748761,  0.26535796,  0.56815550,  1.72189957};
//		double[] x2= {6.329240, 5.033362, 4.645776, 6.584447, 5.664754};
//		System.out.println(t_test(x1,x2));
//		double[] d1= {10,9,10,7}, d2= {0,1,0,3}, t= {0,1,2,3};
//		System.out.println(ca_trend(d1, d2, t));
//		double[] p = {0.1,0.05,0.03,0.04};
//		System.out.println(Fisher_combined(p));
		
//		double[] weight0=set_weights(0);
//		double[] weight1=set_weights(1);
//		for(int k=0;k<weight0.length;k++)System.out.println(weight0[k]+"\t"+weight1[k]);
		
		test_along_freq();
		
	}
}
