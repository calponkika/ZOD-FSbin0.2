function   [OptSeqMeth, N_Outl, Acc]=Optimal_Accuracy(~,~) 
%tic
% % Load any sample size.For me I load Sample size No 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Sample size No 1
%  load Sample_Size22K1.mat
%  M=samplesize1;
% % --  The best method is chain of ZOD and Zscore,Threshold=2.5, Opt=Mean. It performs at 99.35% and has effect of 0.03% The best classical feature scaling method was Zscore..It has accuracy of 99.33%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Sample size No 2
% %  load Sample_Size_33K1.mat
%  M=samplesize2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Sample size No 3
%  load Sample_Size41K1.mat
%   M=samplesize3;
% %   The best method is chain of ZOD(MinMax), Threshold=2.5, Opt=Median. It performs at 99.22% and has effect of 0.61% The best classical feature scaling method was Zscore..It has accuracy of 98.62%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Sample size No 4
%    load Sample_Size45K1
%  M=samplesize4;
% %  The best method is chain of ZOD and MinMax,Threshold=2.9,Opt=Mean. It performs at 99.66% and has effect of 0.26% The best classical feature scaling method was MinMax normalization..It has accuracy of 99.40%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Sample size No 5
    load Sample_Size57K1
   M=samplesize5;
%   The best method is chain of ZOD (Zscore),Threshold=2.7,Opt=Median. It performs at 99.15% and has effect of 0.35% The best classical feature scaling method was Zscore..It has accuracy of 98.80%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Num_Dataset=Data_Conversion(M);
PreFS=Zero_SdtRemoval(Num_Dataset);
[Inputs,Targets]=DataMorph_NNW(PreFS);
A=Inputs;
B=InputSequences(A);
Acc=zeros(1,length(B));
for i=1:length(B)
    InpAcc=B{i};
    Accuracy=NNW_Alg(InpAcc,Targets);
  Acc(i)=Accuracy;
end
OptSeqMeth=Decision(Acc);
N_Outl=Outlier_Detect(A);
%toc
end


