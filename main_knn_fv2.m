
clc
clear all
close all
tic;

load Competition_train.mat

m=2;  %Extract 2m components

%% Downsample

 for i = 1:size(X,1)
   for j = 1:size(X,2)
      
      X_ds(i,j,:) = downsample(squeeze(X(i,j,:)),10);
              
   end    
end

%% Filtering

order = 4;   %Closest even number to 5
fs = 100;
lowFreq = 9;
hiFreq = 20;
X_filt = bpf(X_ds,order,fs,lowFreq,hiFreq);


%% Feature Extraction - CSP PCA

clear X
X = X_filt;

subVal = 278;
X_sub = (X(1:subVal,:,:));
Y_sub = Y(1:subVal);

ind_neg1 = find(Y_sub<0);
ind_pos1 = find(Y_sub>0);

d2 = size(squeeze(X_sub(1,:,:)),1);
d3 = size(squeeze(X_sub(1,:,:)),2);

R1 = zeros(d2,d2);
R2 = zeros(d2,d2);

for i1 = 1:length(ind_neg1)

    dat1 = squeeze(X_sub(ind_neg1(i1),:,:));
    temp1 = (dat1)*(dat1)';
    temp1 = temp1/trace(temp1);
    R1 = R1 + temp1;
        
end

R1 = R1/length(ind_neg1);

for i2 = 1:length(ind_pos1)

    dat2 = squeeze(X_sub(ind_pos1(i2),:,:));
    temp1 = (dat2)*(dat2)';
    temp1 = temp1/trace(temp1);
    R2 = R2 + temp1;
        
end

R2 = R2/length(ind_pos1);

% Ramoser equation (2)
Rsum = R1+R2;

% Find Eigenvalues and Eigenvectors of RC
% Sort eigenvalues in descending order
[EVecsum,EValsum] = eig(Rsum);
[EValsum,ind] = sort(diag(EValsum),'descend');
EVecsum = EVecsum(:,ind);

% Find Whitening Transformation Matrix - Ramoser Equation (3)
W = sqrt(pinv(diag(EValsum))) * EVecsum';

% Whiten Data Using Whiting Transform - Ramoser Equation (4)
S1 = W * R1 * W';
S2 = W * R2 * W';

% Ramoser equation (5)
%[U1,Psi1] = eig(S1);
%[U2,Psi2] = eig(S2);

%generalized eigenvectors/values
[B,D] = eig(S1,S2);

% Simultanous diagonalization
% Should be equivalent to [B,D]=eig(S1);

% verify algorithim
%disp('test1:Psi1+Psi2=I')
%Psi1+Psi2

% sort ascending by default
%[Psi1,ind] = sort(diag(Psi1)); U1 = U1(:,ind);
%[Psi2,ind] = sort(diag(Psi2)); U2 = U2(:,ind);
[D,ind]=sort(diag(D));
B=B(:,ind);

% Resulting Projection Matrix-these are the spatial filter coefficients
projn_mat = B'*W;

for ctr = 1:size(projn_mat,2)
    projn_mat(:,ctr) = projn_mat(:,ctr)/norm(projn_mat(:,ctr));
end

% store_features1 = zeros((size(X_sub,2)*size(X_sub,3)),subVal);

%% Training

row_temp_proj = size(projn_mat,1);

fv = zeros(2*m,1);

% X_best = X_sub(:,:,64:300);

for ind_proj = 1:size(X_sub,1)
 
    temp_proj = projn_mat * squeeze(X_sub(ind_proj,:,:));
    
    
for indr = 1:m
temp_projm(indr,:) = temp_proj(indr,:);
end
dec_ctr = 0;
for indr = m+1:2*m
temp_projm(indr,:) = temp_proj(row_temp_proj-dec_ctr,:);
dec_ctr = dec_ctr + 1;
end
sum_var = 0;
for var_ctr = 1:2*m
sum_var = sum_var + var(temp_projm(var_ctr,:));
end

for ind_fv = 1:2*m
fv(ind_fv) = log(var(temp_projm(ind_fv,:))/sum_var);
end
% pause;
fv_mat(ind_proj,:) = fv;
% pause;
end

train_data2 = fv_mat;
label_train = Y_sub;
% pause;
% mdl = ClassificationKNN.fit(train_data2,label_train);
mdl = ClassificationKNN.fit(train_data2,label_train,'NumNeighbors',3); 

%% ------------------ Testing---------------------------------------

clear X
load Competition_test.mat
true_labels = dlmread('true_labels.txt');

%% Downsample

 for i = 1:size(X,1)
   for j = 1:size(X,2)
      
      X_ds_test(i,j,:) = downsample(squeeze(X(i,j,:)),10);
              
   end    
 end
 
clear X
 %%

 X = bpf(X_ds_test,order,fs,lowFreq,hiFreq);

score = 0;
%  test_ind = input('Enter the sample number to be classified (201 - 278): ');

test_ind = 1 : 100;  %For classification accuracy - 92 %
% test_ind = 1 : 278;  %For training classification accuracy - 99.07%
% test_ind = 201 : 278;


for i = 1:length(test_ind)
    
test_data = projn_mat * squeeze(X(test_ind(i),:,:));
row_test_proj = size(test_data,1);


for indr = 1:m
test_projm(indr,:) = test_data(indr,:);
end
dec_ctr = 0;
for indr = m+1:2*m
test_projm(indr,:) = test_data(row_test_proj-dec_ctr,:);
dec_ctr = dec_ctr + 1;
end
sum_var = 0;
for var_ctr = 1:2*m
sum_var = sum_var + var(test_projm(var_ctr,:));
end

for ind_fv = 1:2*m
test_fv(ind_fv) = log(var(test_projm(ind_fv,:))/sum_var);   
end
% pause;
test_vec = test_fv;
% i

% disp('Label: ');
label = predict(mdl,test_vec);


% disp('True Val = ');
% disp(Y(test_ind));

if((true_labels(test_ind(i))) == label)
% if(Y(test_ind(i)) == label)
%     disp('inside if');
    score = score + 1;
end

end

perc_correct = (score/length(test_ind))*100
elapsedTime = toc

beep on;
beep;
pause(2);
beep;
pause(2);
beep;