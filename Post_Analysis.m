clc
clear

load('Muts.mat')

load('EXTER_TEST_RESULTS_WGCN_CPROP.mat')

YPRED=squeeze(YPRED);
Data=readtable('External_Data');
FC=string(Data.FoldChange);
ACT=Data.Activity;
WT=Data.WildType;
STRAIN=string(Data.Strain);
AT=string(Data.AssayType);
DC=string(Data.Drug_Compound);

for cvcase=1:5
IND=find(~isnan(ACT) & ~isnan(WT));
ind=setdiff(1:2667,1+V_indices{1,cvcase}); 
IND=IND(1+V_indices{1,cvcase});
Y_PRED=YPRED(cvcase,:);
Y_PRED=Y_PRED(1 ,1+V_indices{1,cvcase});


for i=1:length(IND)
if FC(IND(i))=='No'
OUT(i)=ACT(IND(i))./WT(IND(i));
else
OUT(i)=ACT(IND(i));
end
end
STR=STRAIN(IND);

W=Muts;
INP=zeros(536,length(IND));
for i=1:length(IND)
S=string(STR(i));
S=convertStringsToChars(S);
[V]=str_char_improved(S);
for k=1:length(V)
for j=1:536
if convertCharsToStrings(W{j})==convertCharsToStrings(V{k})           
INP(j,i)=1;
end
end
end


end


FOUT=log10(OUT');
S=0;
for j1=1:1
S=S+Y_PRED(j1,:);
end
res=S/1;

figure (1)
scatter(FOUT,res)

INP=INP';
SI=unique(INP,'rows');


for j=1:length(IND)
for i=1:size(SI,1)

if norm(INP(j,:)-SI(i,:))==0

    GR(j,1)=i;

end

end
end

w=0;
p=0;
for i=1:size(SI,1)

    k=find(GR==i);

    if length(k)>=2
p=p+1;

S=0;
for j1=1:1
S=S+Y_PRED(j1,k);
end
RRES=S/1;    
ROUT=FOUT(k);

Z=nchoosek(1:length(ROUT),2);
f=0;
for g=1:size(Z,1)
if abs(ROUT(Z(g,1))-ROUT(Z(g,2)))>0
    f=f+1;

    CRES{p}(f)=RRES(Z(g,1))-RRES(Z(g,2));
    CERES{p}(f)=double((ROUT(Z(g,1))-ROUT(Z(g,2)))>0);
    
    CERES_VAL{p}(f)=double((ROUT(Z(g,1))-ROUT(Z(g,2))));
    WM{p}(1,f)=k(Z(g,1));
    WM{p}(2,f)=k(Z(g,2));

    CRESR{p}(1,f)=RRES(Z(g,1));
    CRESR{p}(2,f)=RRES(Z(g,2));
    CERESR{p}(1,f)=ROUT(Z(g,1));
    CERESR{p}(2,f)=ROUT(Z(g,2));
end
end

if f~=0

B2=CRES{p}';
ObsInt=CERES{p}';
Obs={};
for ii=1:length(ObsInt)
Obs{ii}=num2str(ObsInt(ii));
end

else

    p=p-1;


end

    end
end

B2=[];
ObsInt=[];
WML=[];
CVAL=[];
for i=1:p
B2=[B2;CRES{i}'];
ObsInt=[ObsInt;CERES{i}'];

WML=[WML,WM{i}];
CVAL=[CVAL,CERES_VAL{i}];
end
B2=(B2-min(B2))./(max(B2)-min(B2));
CVAL2=abs(CVAL);

Obs={};
for ii=1:length(ObsInt)
Obs{ii}=num2str(ObsInt(ii));
end
%For ROC curve
 [X,Y,T,AUC] = perfcurve(Obs,B2,'1');
%%For PR curve
% [X,Y,~,AUC] = perfcurve(Obs,B2,'1','xCrit','reca','yCrit','prec');
figure (3)
plot(X,Y,'LineWidth',1)
title(num2str(AUC))
AUC_Ranking_Cl(cvcase)=AUC;
B2=[];
for i=1:p
B2=[B2;CRES{i}'];
end

YP(B2>=0)=1;
YP(B2<0)=0;
YT=ObsInt';

[Accuracy_Ranking_Cl(cvcase),Sensitivity_Ranking_Cl(cvcase),Specifity_Ranking_Cl(cvcase)]=class_perform(YP,YT);

TRS=find(YP==YT);
WML=WML(:,TRS);
CVAL=CVAL(:,TRS);

LT=FOUT';
B2=res;
ObsInt=[];
ObsInt(LT>log10(3))=1;
ObsInt(LT<=log10(3))=0;
B2=(B2-min(B2))./(max(B2)-min(B2));

Obs={};
for ii=1:length(ObsInt)
Obs{ii}=num2str(ObsInt(ii));
end
[X,Y,T,AUC] = perfcurve(Obs,B2,'1');
% [X,Y,~,AUC] = perfcurve(Obs,B2,'1','xCrit','reca','yCrit','prec');
AUC_Resistance_Cl(cvcase)=AUC;

figure (4)
plot(X,Y,'LineWidth',1)
title(num2str(AUC))

YPP(res>=log10(3))=1;
YPP(res<log10(3))=0;
[Accuracy_Resistance_Cl(cvcase),Sensitivity_Resistance_Cl(cvcase),Specifity_Resistance_Cl(cvcase)]=class_perform(YPP,ObsInt);

ACC=abs(ROUT-res);

NACC=(ACC-min(ACC))./(max(ACC)-min(ACC));
T=table();
T.Drug_Compound=Data.Drug_Compound(IND);
 T.Smiles=Data.SMILES(IND);
  T.Strain=Data.Strain(IND);

 T.MAE=ACC';
 T.NMAE=NACC';
 T.Experimental_Log_Fold_Change=FOUT;
 T.Prediction_Log_Fold_Change=res';
%   
TT=Data(IND,:);

DOUT=[];
POUT=[];
for i=1:8
DOUT=[DOUT,CERESR{i}];
POUT=[POUT,CRESR{i}];
end

figure (5)
subplot(1,2,1)
scatter(DOUT(1,:),DOUT(2,:))
subplot(1,2,2)
scatter(POUT(1,:),POUT(2,:))

corrmat=corrcoef(FOUT,res);
corr(cvcase)=corrmat(1,2);

ytrue = FOUT;
ypred = res';

msemat = mean((ytrue - ypred).^2);
mse(cvcase)=msemat;

mean_true = mean(ytrue);
tss = sum((ytrue - mean_true).^2);
rss = sum((ytrue - ypred).^2);
r_squared(cvcase) = 1 - (rss / tss);

end