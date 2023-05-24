% The raw dataset partition for diagnosis and detection
clc
close all
clear
%% public parameters
% detection: normal(0) abnormal(1)
% diagnosis: normal(0) vsc_delay(f1) vsc_delay(f2) vsc_delay(f3)
% vsc_delay(f4) vsc_delay(f5) vsc_delay(f6) dcdc_fault(f7, contain_5)
% dcdc_fault(f8, doe not contain_5)

% partition parameters, decided by the user
% window size 
window_size = 100;

% step size when window moves
s_step = 0.2 * 100;

% sample frequency, stuff for fft 
fs = 1000;
df = fs / (window_size - 1);
f = (0 : window_size - 1) * df;

% specify the source folder: normal(1834),dcdc_fault(1921), vsc_delay(180)
%% create specific dataset
current_directory = strcat('w',int2str(window_size),'_final_dataset');
mkdir(current_directory)
mkdir(current_directory,'fault_detection')
mkdir(current_directory,'fault_diagnosis')
mkdir(strcat(current_directory,'/fault_diagnosis'),'deep_learning')
mkdir(strcat(current_directory,'/fault_diagnosis'),'feature')

mkdir(strcat(current_directory,'/fault_detection'),'deep_learning')
mkdir(strcat(current_directory,'/fault_detection'),'feature')

mkdir(strcat(current_directory, '/fault_detection/deep_learning'), 'Normal')
mkdir(strcat(current_directory, '/fault_detection/deep_learning'), 'Abnormal')

mkdir(strcat(current_directory, '/fault_detection/feature'), 'Normal')
mkdir(strcat(current_directory, '/fault_detection/feature'), 'Abnormal')

mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Normal')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_1')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_2')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_3')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_4')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_5')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_6')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_7')
mkdir(strcat(current_directory, '/fault_diagnosis/deep_learning'), 'Fault_8')

mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Normal')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_1')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_2')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_3')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_4')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_5')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_6')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_7')
mkdir(strcat(current_directory, '/fault_diagnosis/feature'), 'Fault_8')


%% vsc_dealy data partition
% input by user
num_of_file = 180;
raw_files = dir('./VSC_delay/*.csv');

for i=1:num_of_file
    ref_matrix = csvread(strcat('./VSC_delay/', raw_files(i).name));
    Va = ref_matrix(:,1);
    Vb = ref_matrix(:,2);
    Vc = ref_matrix(:,3); 
    
    Ia = ref_matrix(:,4);
    Ib = ref_matrix(:,5);
    Ic = ref_matrix(:,6);
    
    %envelope for calculating magnitude
    [up_Va, down_Va] = envelope(Va);
    [up_Vb, down_Vb] = envelope(Vb);
    [up_Vc, down_Vc] = envelope(Vc);
    
    [up_Ia, down_Ia] = envelope(Ia);
    [up_Ib, down_Ib] = envelope(Ib);
    [up_Ic, down_Ic] = envelope(Ic);
    
    % feature matrix for machine learning methods (ANN, SVM, KNN, DT)
    feature_matrix_normal = [];
    feature_matrix_fault = [];
    
    % When diagnosis, determine which attack type does the file belong to.
    filename_diagnosis_dl_normal = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Normal/');
    filename_diagnosis_ft_normal = strcat('./', current_directory, '/fault_diagnosis/feature/Normal/');
    fault_type = str2double(raw_files(i).name(strfind(raw_files(i).name,'.cs') - 1));
    switch fault_type
        case 1
            filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_1/');
            filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_1/');
        case 2
            filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_2/');
            filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_2/');
        case 3
            filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_3/');
            filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_3/');
        case 4
            filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_4/');
            filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_4/');
        case 5
            filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_5/');
            filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_5/');
        case 6
            filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_6/');
            filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_6/');
    end
    
    filename_detection_dl_normal = strcat('./', current_directory, '/fault_detection/deep_learning/Normal/');
    filename_detection_dl_abnormal = strcat('./', current_directory, '/fault_detection/deep_learning/Abnormal/');
    filename_detection_ft_normal = strcat('./', current_directory, '/fault_detection/feature/Normal/');
    filename_detection_ft_abnormal = strcat('./', current_directory, '/fault_detection/feature/Abnormal/');
    
    
    for window_start=1:s_step:500 - window_size + 1
                
        frame_Va = Va(window_start:window_start + window_size - 1);
        frame_Vb = Vb(window_start:window_start + window_size - 1);
        frame_Vc = Vc(window_start:window_start + window_size - 1);
        
        frame_Ia = Ia(window_start:window_start + window_size - 1);
        frame_Ib = Ib(window_start:window_start + window_size - 1);
        frame_Ic = Ic(window_start:window_start + window_size - 1);
        
        % the following are features extracting from raw waveform
        frame_mag_Va = rms(up_Va(window_start:window_start + window_size - 1));
        frame_mag_Vb = rms(up_Vb(window_start:window_start + window_size - 1));
        frame_mag_Vc = rms(up_Vc(window_start:window_start + window_size - 1));
        
        frame_mag_Ia = rms(up_Ia(window_start:window_start + window_size - 1));
        frame_mag_Ib = rms(up_Ib(window_start:window_start + window_size - 1));
        frame_mag_Ic = rms(up_Ic(window_start:window_start + window_size - 1));
        
        
        Y_Va = fft(frame_Va) / window_size * 2;
        Y_Vb = fft(frame_Vb) / window_size * 2;
        Y_Vc = fft(frame_Vc) / window_size * 2;
        
        Y_Ia = fft(frame_Ia) / window_size * 2;
        Y_Ib = fft(frame_Ib) / window_size * 2;
        Y_Ic = fft(frame_Ic) / window_size * 2;
        
        [m_Va, pos_Va] = max(abs(Y_Va(1:window_size / 2)));
        [m_Vb, pos_Vb] = max(abs(Y_Vb(1:window_size / 2)));
        [m_Vc, pos_Vc] = max(abs(Y_Vc(1:window_size / 2)));
        
        
        [m_Ia, pos_Ia] = max(abs(Y_Ia(1:window_size / 2)));
        [m_Ib, pos_Ib] = max(abs(Y_Ib(1:window_size / 2)));
        [m_Ic, pos_Ic] = max(abs(Y_Ic(1:window_size / 2)));
        
        frame_freq_Va = f(pos_Va);
        frame_freq_Vb = f(pos_Vb);
        frame_freq_Vc = f(pos_Vc);
        
        frame_freq_Ia = f(pos_Ia);
        frame_freq_Ib = f(pos_Ib);
        frame_freq_Ic = f(pos_Ic);
        
        ph_Va = unwrap(angle(hilbert(frame_Va)));
        ph_Vb = unwrap(angle(hilbert(frame_Vb)));
        ph_Vc = unwrap(angle(hilbert(frame_Vc)));
        
        ph_Ia = unwrap(angle(hilbert(frame_Ia)));
        ph_Ib = unwrap(angle(hilbert(frame_Ib)));
        ph_Ic = unwrap(angle(hilbert(frame_Ic)));
        
        frame_phV_diff_ab = mean(abs((ph_Va - ph_Vb)*180/pi));
        frame_phV_diff_ac = mean(abs((ph_Va - ph_Vc)*180/pi));
        frame_phV_diff_bc = mean(abs((ph_Vb - ph_Vc)*180/pi));
        
        frame_phI_diff_ab = mean(abs((ph_Ia - ph_Ib)*180/pi));
        frame_phI_diff_ac = mean(abs((ph_Ia - ph_Ic)*180/pi));
        frame_phI_diff_bc = mean(abs((ph_Ib - ph_Ic)*180/pi));
        
%         frame_freq_Va_norm = normalize(frame_freq_Va, 'range', [0,1]);
%         frame_freq_Vb_norm = normalize(frame_freq_Vb, 'range', [0,1]);
%         frame_freq_Vc_norm = normalize(frame_freq_Vc, 'range', [0,1]);
%         frame_freq_Ia_norm = normalize(frame_freq_Ia, 'range', [0,1]);
%         frame_freq_Ib_norm = normalize(frame_freq_Ib, 'range', [0,1]);
%         frame_freq_Ic_norm = normalize(frame_freq_Ic, 'range', [0,1]);
%         frame_mag_Va_norm = normalize(frame_mag_Va, 'range', [0,1]);
%         frame_mag_Vb_norm = normalize(frame_mag_Vb, 'range', [0,1]);
%         frame_mag_Vc_norm = normalize(frame_mag_Vc, 'range', [0,1]);
%         frame_mag_Ia_norm = normalize(frame_mag_Ia, 'range', [0,1]);
%         frame_mag_Ib_norm = normalize(frame_mag_Ib, 'range', [0,1]);
%         frame_mag_Ic_norm = normalize(frame_mag_Ic, 'range', [0,1]);
%         frame_phV_diff_ab_norm = normalize(frame_phV_diff_ab, 'range', [0,1]);
%         frame_phV_diff_ac_norm = normalize(frame_phV_diff_ac, 'range', [0,1]);
%         frame_phV_diff_bc_norm = normalize(frame_phV_diff_bc, 'range', [0,1]);
%         frame_phI_diff_ab_norm = normalize(frame_phI_diff_ab, 'range', [0,1]);
%         frame_phI_diff_ac_norm = normalize(frame_phI_diff_ac, 'range', [0,1]);
%         frame_phI_diff_bc_norm = normalize(frame_phI_diff_bc, 'range', [0,1]);        
        
        frame_deep_matrix = [frame_Va, frame_Vb, frame_Vc, frame_Ia, frame_Ib, frame_Ic];
        frame_feature_matrix = [frame_freq_Va, frame_freq_Vb, frame_freq_Vc, ...
                                frame_freq_Ia, frame_freq_Ib, frame_freq_Ic, ...
                                frame_mag_Va, frame_mag_Vb, frame_mag_Vc, ...
                                frame_mag_Ia, frame_mag_Ib, frame_mag_Ic, ...
                                frame_phV_diff_ab, frame_phV_diff_ac, frame_phV_diff_bc, ...
                                frame_phI_diff_ab, frame_phI_diff_ac, frame_phI_diff_bc];
        
        % abnormal                    
        if window_start + window_size - 1 > 250
            % For deep learning fault detection, go to the abnormal folder
            fname = strcat(filename_detection_dl_abnormal, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For deep learning fault diagnosis
            fname = strcat(filename_dl, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For feature matrix based fault detection and diagnosis
            feature_matrix_fault = [feature_matrix_fault; frame_feature_matrix];
            
        % normal
        else
            % For deep learning normal detection, go to the normal folder
            fname = strcat(filename_detection_dl_normal, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For deep learning fault diagnosis
            fname = strcat(filename_diagnosis_dl_normal, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For feature matrix based fault detection and diagnosis
            feature_matrix_normal = [feature_matrix_normal; frame_feature_matrix];
        end
        
    end
    
     % save feature matrix when normal diagnosis
    fname = strcat(filename_diagnosis_ft_normal, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_normal);

    % diagnosis abnormal
    fname = strcat(filename_ft, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_fault);

    % detecion normal
    fname = strcat(filename_detection_ft_normal, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_normal);

    % detection abnormal
    fname = strcat(filename_detection_ft_abnormal, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_fault);  
end

    % if contain type 5, it's a fault

%% dcdc_fault partition
num_of_file = 1921;
raw_files = dir('./DCDC_fault/*.csv');

for i=1:num_of_file
    ref_matrix = csvread(strcat('./DCDC_fault/', raw_files(i).name));
    Va = ref_matrix(:,1);
    Vb = ref_matrix(:,2);
    Vc = ref_matrix(:,3); 
    
    Ia = ref_matrix(:,4);
    Ib = ref_matrix(:,5);
    Ic = ref_matrix(:,6);
    
    %envelope for calculating magnitude
    [up_Va, down_Va] = envelope(Va);
    [up_Vb, down_Vb] = envelope(Vb);
    [up_Vc, down_Vc] = envelope(Vc);
    
    [up_Ia, down_Ia] = envelope(Ia);
    [up_Ib, down_Ib] = envelope(Ib);
    [up_Ic, down_Ic] = envelope(Ic);
    
    % feature matrix for machine learning methods (ANN, SVM, KNN, DT)
    feature_matrix_normal = [];
    feature_matrix_fault = [];
    
    % When diagnosis, determine which attack type does the file belong to.
    filename_diagnosis_dl_normal = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Normal/');
    filename_diagnosis_ft_normal = strcat('./', current_directory, '/fault_diagnosis/feature/Normal/');
     
    % if contain type 5, it's a fault
    flag = strfind(raw_files(i).name,'k_')+4;
    if contains(raw_files(i).name(flag:flag+3), '5')
        filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_7/');
        filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_7/');
    else
        filename_dl = strcat('./', current_directory, '/fault_diagnosis/deep_learning/Fault_8/');
        filename_ft = strcat('./', current_directory, '/fault_diagnosis/feature/Fault_8/');
    end
    
    filename_detection_dl_normal = strcat('./', current_directory, '/fault_detection/deep_learning/Normal/');
    filename_detection_dl_abnormal = strcat('./', current_directory, '/fault_detection/deep_learning/Abnormal/');
    filename_detection_ft_normal = strcat('./', current_directory, '/fault_detection/feature/Normal/');
    filename_detection_ft_abnormal = strcat('./', current_directory, '/fault_detection/feature/Abnormal/');
    
    
    for window_start=1:s_step:500 - window_size + 1
                
        frame_Va = Va(window_start:window_start + window_size - 1);
        frame_Vb = Vb(window_start:window_start + window_size - 1);
        frame_Vc = Vc(window_start:window_start + window_size - 1);
        
        frame_Ia = Ia(window_start:window_start + window_size - 1);
        frame_Ib = Ib(window_start:window_start + window_size - 1);
        frame_Ic = Ic(window_start:window_start + window_size - 1);
        
        % the following are features extracting from raw waveform
        frame_mag_Va = rms(up_Va(window_start:window_start + window_size - 1));
        frame_mag_Vb = rms(up_Vb(window_start:window_start + window_size - 1));
        frame_mag_Vc = rms(up_Vc(window_start:window_start + window_size - 1));
        
        frame_mag_Ia = rms(up_Ia(window_start:window_start + window_size - 1));
        frame_mag_Ib = rms(up_Ib(window_start:window_start + window_size - 1));
        frame_mag_Ic = rms(up_Ic(window_start:window_start + window_size - 1));
        
        
        Y_Va = fft(frame_Va) / window_size * 2;
        Y_Vb = fft(frame_Vb) / window_size * 2;
        Y_Vc = fft(frame_Vc) / window_size * 2;
        
        Y_Ia = fft(frame_Ia) / window_size * 2;
        Y_Ib = fft(frame_Ib) / window_size * 2;
        Y_Ic = fft(frame_Ic) / window_size * 2;
        
        [m_Va, pos_Va] = max(abs(Y_Va(1:window_size / 2)));
        [m_Vb, pos_Vb] = max(abs(Y_Vb(1:window_size / 2)));
        [m_Vc, pos_Vc] = max(abs(Y_Vc(1:window_size / 2)));
        
        
        [m_Ia, pos_Ia] = max(abs(Y_Ia(1:window_size / 2)));
        [m_Ib, pos_Ib] = max(abs(Y_Ib(1:window_size / 2)));
        [m_Ic, pos_Ic] = max(abs(Y_Ic(1:window_size / 2)));
        
        frame_freq_Va = f(pos_Va);
        frame_freq_Vb = f(pos_Vb);
        frame_freq_Vc = f(pos_Vc);
        
        frame_freq_Ia = f(pos_Ia);
        frame_freq_Ib = f(pos_Ib);
        frame_freq_Ic = f(pos_Ic);
        
        ph_Va = unwrap(angle(hilbert(frame_Va)));
        ph_Vb = unwrap(angle(hilbert(frame_Vb)));
        ph_Vc = unwrap(angle(hilbert(frame_Vc)));
        
        ph_Ia = unwrap(angle(hilbert(frame_Ia)));
        ph_Ib = unwrap(angle(hilbert(frame_Ib)));
        ph_Ic = unwrap(angle(hilbert(frame_Ic)));
        
        frame_phV_diff_ab = mean(abs((ph_Va - ph_Vb)*180/pi));
        frame_phV_diff_ac = mean(abs((ph_Va - ph_Vc)*180/pi));
        frame_phV_diff_bc = mean(abs((ph_Vb - ph_Vc)*180/pi));
        
        frame_phI_diff_ab = mean(abs((ph_Ia - ph_Ib)*180/pi));
        frame_phI_diff_ac = mean(abs((ph_Ia - ph_Ic)*180/pi));
        frame_phI_diff_bc = mean(abs((ph_Ib - ph_Ic)*180/pi));
        
        frame_deep_matrix = [frame_Va, frame_Vb, frame_Vc, frame_Ia, frame_Ib, frame_Ic];
        frame_feature_matrix = [frame_freq_Va, frame_freq_Vb, frame_freq_Vc, ...
                                frame_freq_Ia, frame_freq_Ib, frame_freq_Ic, ...
                                frame_mag_Va, frame_mag_Vb, frame_mag_Vc, ...
                                frame_mag_Ia, frame_mag_Ib, frame_mag_Ic, ...
                                frame_phV_diff_ab, frame_phV_diff_ac, frame_phV_diff_bc, ...
                                frame_phI_diff_ab, frame_phI_diff_ac, frame_phI_diff_bc];
                            
        
        % abnormal                    
        if window_start + window_size - 1 > 250
            % For deep learning fault detection, go to the abnormal folder
            fname = strcat(filename_detection_dl_abnormal, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For deep learning fault diagnosis
            fname = strcat(filename_dl, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For feature matrix based fault detection and diagnosis
            feature_matrix_fault = [feature_matrix_fault; frame_feature_matrix];
            
        % normal
        else
            % For deep learning normal detection, go to the normal folder
            fname = strcat(filename_detection_dl_normal, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For deep learning fault diagnosis
            fname = strcat(filename_diagnosis_dl_normal, int2str(window_start),'__',raw_files(i).name);
            csvwrite(fname, frame_deep_matrix);
            
            % For feature matrix based fault detection and diagnosis
            feature_matrix_normal = [feature_matrix_normal; frame_feature_matrix];
        end
        
    end
    
     % save feature matrix when normal diagnosis
    fname = strcat(filename_diagnosis_ft_normal, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_normal);

    % diagnosis abnormal
    fname = strcat(filename_ft,'feature__',raw_files(i).name);
    csvwrite(fname, feature_matrix_fault);

    % detecion normal
    fname = strcat(filename_detection_ft_normal,'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_normal);

    % detection abnormal
    fname = strcat(filename_detection_ft_abnormal,'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_fault);  
end
%% normal data partition
num_of_file = 1834;
raw_files = dir('./normal/*.csv');

for i=1:num_of_file
    ref_matrix = csvread(strcat('./normal/', raw_files(i).name));
    Va = ref_matrix(:,1);
    Vb = ref_matrix(:,2);
    Vc = ref_matrix(:,3); 
    
    Ia = ref_matrix(:,4);
    Ib = ref_matrix(:,5);
    Ic = ref_matrix(:,6);
    
    %envelope for calculating magnitude
    [up_Va, down_Va] = envelope(Va);
    [up_Vb, down_Vb] = envelope(Vb);
    [up_Vc, down_Vc] = envelope(Vc);
    
    [up_Ia, down_Ia] = envelope(Ia);
    [up_Ib, down_Ib] = envelope(Ib);
    [up_Ic, down_Ic] = envelope(Ic);
    
    % feature matrix for machine learning methods (ANN, SVM, KNN, DT)
    feature_matrix_normal = [];
    
    % When diagnosis, determine which attack type does the file belong to.
    filename_diagnosis_dl_normal = strcat('./', current_directory,'/fault_diagnosis/deep_learning/Normal/');
    filename_diagnosis_ft_normal = strcat('./', current_directory,'/fault_diagnosis/feature/Normal/');

    filename_detection_dl_normal = strcat('./', current_directory,'/fault_detection/deep_learning/Normal/');
    filename_detection_ft_normal = strcat('./', current_directory,'/fault_detection/feature/Normal/');
    
    
    for window_start=1:s_step:500 - window_size + 1
                
        frame_Va = Va(window_start:window_start + window_size - 1);
        frame_Vb = Vb(window_start:window_start + window_size - 1);
        frame_Vc = Vc(window_start:window_start + window_size - 1);
        
        frame_Ia = Ia(window_start:window_start + window_size - 1);
        frame_Ib = Ib(window_start:window_start + window_size - 1);
        frame_Ic = Ic(window_start:window_start + window_size - 1);
        
        % the following are features extracting from raw waveform
        frame_mag_Va = rms(up_Va(window_start:window_start + window_size - 1));
        frame_mag_Vb = rms(up_Vb(window_start:window_start + window_size - 1));
        frame_mag_Vc = rms(up_Vc(window_start:window_start + window_size - 1));
        
        frame_mag_Ia = rms(up_Ia(window_start:window_start + window_size - 1));
        frame_mag_Ib = rms(up_Ib(window_start:window_start + window_size - 1));
        frame_mag_Ic = rms(up_Ic(window_start:window_start + window_size - 1));
        
        
        Y_Va = fft(frame_Va) / window_size * 2;
        Y_Vb = fft(frame_Vb) / window_size * 2;
        Y_Vc = fft(frame_Vc) / window_size * 2;
        
        Y_Ia = fft(frame_Ia) / window_size * 2;
        Y_Ib = fft(frame_Ib) / window_size * 2;
        Y_Ic = fft(frame_Ic) / window_size * 2;
        
        [m_Va, pos_Va] = max(abs(Y_Va(1:window_size / 2)));
        [m_Vb, pos_Vb] = max(abs(Y_Vb(1:window_size / 2)));
        [m_Vc, pos_Vc] = max(abs(Y_Vc(1:window_size / 2)));
        
        
        [m_Ia, pos_Ia] = max(abs(Y_Ia(1:window_size / 2)));
        [m_Ib, pos_Ib] = max(abs(Y_Ib(1:window_size / 2)));
        [m_Ic, pos_Ic] = max(abs(Y_Ic(1:window_size / 2)));
        
        frame_freq_Va = f(pos_Va);
        frame_freq_Vb = f(pos_Vb);
        frame_freq_Vc = f(pos_Vc);
        
        frame_freq_Ia = f(pos_Ia);
        frame_freq_Ib = f(pos_Ib);
        frame_freq_Ic = f(pos_Ic);
        
        ph_Va = unwrap(angle(hilbert(frame_Va)));
        ph_Vb = unwrap(angle(hilbert(frame_Vb)));
        ph_Vc = unwrap(angle(hilbert(frame_Vc)));
        
        ph_Ia = unwrap(angle(hilbert(frame_Ia)));
        ph_Ib = unwrap(angle(hilbert(frame_Ib)));
        ph_Ic = unwrap(angle(hilbert(frame_Ic)));
        
        frame_phV_diff_ab = mean(abs((ph_Va - ph_Vb)*180/pi));
        frame_phV_diff_ac = mean(abs((ph_Va - ph_Vc)*180/pi));
        frame_phV_diff_bc = mean(abs((ph_Vb - ph_Vc)*180/pi));
        
        frame_phI_diff_ab = mean(abs((ph_Ia - ph_Ib)*180/pi));
        frame_phI_diff_ac = mean(abs((ph_Ia - ph_Ic)*180/pi));
        frame_phI_diff_bc = mean(abs((ph_Ib - ph_Ic)*180/pi));
        
        
        frame_deep_matrix = [frame_Va, frame_Vb, frame_Vc, frame_Ia, frame_Ib, frame_Ic];
        frame_feature_matrix = [frame_freq_Va, frame_freq_Vb, frame_freq_Vc, ...
                                frame_freq_Ia, frame_freq_Ib, frame_freq_Ic, ...
                                frame_mag_Va, frame_mag_Vb, frame_mag_Vc, ...
                                frame_mag_Ia, frame_mag_Ib, frame_mag_Ic, ...
                                frame_phV_diff_ab, frame_phV_diff_ac, frame_phV_diff_bc, ...
                                frame_phI_diff_ab, frame_phI_diff_ac, frame_phI_diff_bc];
        

        % normal
       
        % For deep learning normal detection, go to the normal folder
        fname = strcat(filename_detection_dl_normal, int2str(window_start),'__',raw_files(i).name);
        csvwrite(fname, frame_deep_matrix);

        % For deep learning fault diagnosis
        fname = strcat(filename_diagnosis_dl_normal, int2str(window_start),'__',raw_files(i).name);
        csvwrite(fname, frame_deep_matrix);

        % For feature matrix based fault detection and diagnosis
        feature_matrix_normal = [feature_matrix_normal; frame_feature_matrix];
        
        
    end
    
     % save feature matrix when normal diagnosis
    fname = strcat(filename_diagnosis_ft_normal, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_normal);

    % detecion normal
    fname = strcat(filename_detection_ft_normal, 'feature__', raw_files(i).name);
    csvwrite(fname, feature_matrix_normal);

end