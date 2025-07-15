# HRV-KSS-Classification

# Pre-Process Module Description

Script Names Description 
------------------------

"A_" -> Analysis or Visualization files
"F_" -> Filter or Format datasets
"M_" -> Merge of the processed datasets
"HRV_" -> Files related with ECG files and with HRV features

Execution Order 
---------------

1. Dataset Analysis
A_pvt_data.py -> basic analysis of the PVT files
A_cardioID_data.py -> basic analysis of the simulators cardioID folder
A_sim_data.py -> basic analysis of the data from the sim
A_EDF_stats.py -> basic analysis of EDF files

2. Compute the duration of each ECG
A_edf_duration.py -> Create a .csv file with all the original ECG durations

3. Compare the values of the Marker signal from the EDFs files with the vitaport_value from sim data
A_compare_Marker_and_Sim_Vitaport.py -> file to compare the Marker values from EDF files and the vitaport feature from the Sim Dataset(Result: when vitaport feature = 0 , Marker = 34)

4. Filter ECG with the marker signal
F_ECG_Marker.py -> Filter all EDFs EGC signals with the marker signal = 34 and save new filtered EGCs
F_ECG_durations.py -> Create a .csv file with all the filtered ECG durations

5. Filter the Sim durations. Merge sim1 and sim2 and select only wanted features.
F_Sim_duration.py -> Compute the real durations of the sim by detecting missing instances (threshold of 0.5s)
F_Sim1_Sim2.py -> Merge sim1 and sim2. Filter sim data with vitaport_value = 0 (to match with Marker from EDFs). Select only wanted features (timer [s], kss_answer, Filename)

6. Extract HRV features
HRV_time_domain.ipynb -> extract HRV time domain features from filtered signals
HRV_freq_domain.ipynb -> extract HRV frequency domain features from filtered signals

7. Merge Time and Frequency Domain Features and Merge with the Sim data
M_HRV.py -> merge HRV time domain datasets with frequency domain datasets to create a combined dataset
M_HRV_Sim.py -> merge filtered sim datasets with filtered HRV datasets on the same filenames and in 2 or 5 min intervals

8. Analysis of final pre-process results
A_kss_time.ipynb -> Visualize Evolution of KSS during the time
A_HRV_Sim_durations.ipynb -> visulaization of the durations diferences from Sim, ECGs, filtered Sim and Filtered ECGs 
