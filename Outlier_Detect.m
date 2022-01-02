function [L] =Outlier_Detect(B)
A=zscore(B);
K= abs(A);
T=[2.5,2.7,2.9,3.1,3.3,3.5];
 fprintf('The number of detected outliers in your inputs are listed below\n')
 fprintf('%2s%12s%16s\n','Sno','Threshold','Nbr of outliers');
fprintf ('--------------------------------\n');
 for i=1:length(T)
 T1=T(:,i); 
 Y=find( K > T1);
 L=length(Y);
fprintf ('%2d%12.1f%14d\n',i,T1,L);
 end
end
 