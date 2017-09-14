function [] = genRawData_func(start_id,end_id,part_id)

SAVEPATH='/home/hyli/Data/InternData/trainDB_lps/RawData_Part';
SAVEPATH=[SAVEPATH,num2str(part_id),'.mat'];
noisebase='/home/hyli/Data/InternData/noiseDB/noise_';
cleanbase='/home/hyli/Data/InternData/train_clean/sentence_';



sent_id=start_id;
snr=-5:5:20;

data=[];label=[];nat=[];
clean_spec=[];noisy_spec=[]; sentence_id=[0]; 
    while sent_id<=end_id
        in=1;
        [clean,~]=audioread([cleanbase,num2str(sent_id),'.wav']);
        clean = clean.*32768;
        while in<=35
            [noise,~]=audioread([noisebase,num2str(in),'.wav']);
            noise = noise.*32768;
            rand_snr=snr(randi(6,[1,1]));
            [mix,~]=addnoise(clean,noise,rand_snr);
            s=clean;
            n=mix-s;
            S=2.*log(abs(STFT(s))'+1e-8);
            N=2.*log(abs(STFT(n))'+1e-8);
            M=2.*log(abs(STFT(mix))'+1e-8);
            label=[label;[S,N]]; % clean lps and noise lps
            
            [m,~]=size(M);
            nat_tmp=repmat(mean(M(1:8,:)),[m,1]);
            data=[data;M];
            nat=[nat;nat_tmp];
            in=in+1;
            sentence_id=[sentence_id;sentence_id(end)+m];
        end
        fprintf('sentence id %d finished\n', sent_id);
        sent_id=sent_id+1;
    end

data=single(data);
label=single(label);
nat=single(nat);


save(SAVEPATH,'data','label','nat','sentence_id','-v7.3');
end

