function D =IsOutlier_2p9_Median(B)

A=zscore(B);
[ra,ca]=size(A);
Median_1=zeros(1,ca);
 D=zeros(ra,ca);
 K=zeros(ra,ca);
for i=1:ca
Median1=median(A(:,i));
Median_1(i)=Median1;
end
for i=1:ca
 for j=1:ra
if abs(A(j,i))>2.9   
   K(j,i)=Median_1(1,i); 
    
end
D=K;
 end
 end
 end