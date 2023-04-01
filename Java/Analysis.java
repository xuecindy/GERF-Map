package sound_wave;

import java.io.File;

public class Analysis {

	public static String root_folder="/Users/fan/Dropbox/Zhou/ScienceFair/Data2023-02-17/";
	public final static String[] IDs= {"16","18","19","20","21","22","23","25","26","28",
			"29","31","32","36","37","38","39","40","43"};
	public final static String[] names= {"freq_BF", "freq_BW", "freq_noise", 
								"time_latency","time_50duration","time_rising_slope10", "time_noise",
								"amp_threshold", "amp_DR_x", "amp_DR_slope", "amp_noise"};
	
	public static void generate_data_for_plots() {		
		try {			
			// Generating post and control files:			 
			String prefix="Control";
			for(int ID_index=0; ID_index<IDs.length; ID_index++) {
				String raw_txt= root_folder+"txt/"+prefix+IDs[ID_index]+".qua.txt";
				String working_folder=root_folder+"sample_"+IDs[ID_index]+"/";
				File working_folder_file=new File(working_folder);
				if(!working_folder_file.exists())working_folder_file.mkdir();
				String formated= working_folder+prefix+IDs[ID_index]+".formated.txt";
				String smoothed= working_folder+prefix+IDs[ID_index]+".smoothed.txt";
				String warning_file = working_folder+prefix+IDs[ID_index]+".warning.txt"; 
				String contour_folder=working_folder+prefix+IDs[ID_index]+"_contour_files/";
				File contour_folder_file=new File(contour_folder);
				if(!contour_folder_file.exists())contour_folder_file.mkdir();
				int level=1;
				ThreeD_Data data=new ThreeD_Data(raw_txt, "byFA");
				data.write2file(formated);
				ThreeD_Data data_again=new ThreeD_Data(formated, "byT");
				data_again.smoothing(level);
				data_again.write2file(smoothed);
				ThreeD_Data data_smoothed=new ThreeD_Data(smoothed, "byT");
				float amp_turn_prop=(float)0.9, half_dur_proportion=(float)0.5, rs10_proportion=(float)0.1, continuous_cutoff=(float)0.6;
				ContourExtraction extracted = new ContourExtraction(data_smoothed, amp_turn_prop, half_dur_proportion, 
						rs10_proportion, continuous_cutoff,
						warning_file, contour_folder);				
			}
			prefix="Post";
			for(int ID_index=0; ID_index<IDs.length; ID_index++) {
				String raw_txt= root_folder+"txt/"+prefix+IDs[ID_index]+".qua.txt";
				String working_folder=root_folder+"sample_"+IDs[ID_index]+"/";
				File working_folder_file=new File(working_folder);
				if(!working_folder_file.exists())working_folder_file.mkdir();
				String formated= working_folder+prefix+IDs[ID_index]+".formated.txt";
				String smoothed= working_folder+prefix+IDs[ID_index]+".smoothed.txt";
				String warning_file = working_folder+prefix+IDs[ID_index]+".warning.txt"; 
				String contour_folder=working_folder+prefix+IDs[ID_index]+"_contour_files/";
				File contour_folder_file=new File(contour_folder);
				if(!contour_folder_file.exists())contour_folder_file.mkdir();
				int level=1;
				ThreeD_Data data=new ThreeD_Data(raw_txt, "byFA");
				data.write2file(formated);
				ThreeD_Data data_again=new ThreeD_Data(formated, "byT");
				data_again.smoothing(level);
				data_again.write2file(smoothed);
				ThreeD_Data data_smoothed=new ThreeD_Data(smoothed, "byT");
				float amp_turn_prop=(float)0.9, half_dur_proportion=(float)0.5, rs10_proportion=(float)0.1, continuous_cutoff=(float)0.6;
				ContourExtraction extracted = new ContourExtraction(data_smoothed, amp_turn_prop, half_dur_proportion, 
						rs10_proportion, continuous_cutoff,
						warning_file, contour_folder);
			}
			// Calculating the difference 
			for(int ID_index=0; ID_index<IDs.length; ID_index++) {
				String working_folder=root_folder+"sample_"+IDs[ID_index]+"/";
				
				String pre_contour_folder=working_folder+"Control"+IDs[ID_index]+"_contour_files/";
				String post_contour_folder=working_folder+"Post"+IDs[ID_index]+"_contour_files/";
				String diff_contour_folder=working_folder+"Diff"+IDs[ID_index]+"_contour_files/";
				File diff_contour_folder_file=new File(diff_contour_folder);
				if(!diff_contour_folder_file.exists())diff_contour_folder_file.mkdir();				
				for(int k=0;k<names.length;k++) {
					ContourExtraction.substract(pre_contour_folder+names[k]+".csv", 
							post_contour_folder+names[k]+".csv", diff_contour_folder+names[k]+".csv");
				}
			} 
		}catch(Exception e) {e.printStackTrace();}
	}
	
	public static void main(String[] args) {
		generate_data_for_plots();
	}

}
