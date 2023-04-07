clear all
close all
clc

addpath Data
addpath Algorithms
addpath Tools

warning off

%% Load MNIST dataset 

% subset of MNIST with only 0 and 1
load('Data/dataset_subMNIST.mat')
% full MNIST dataset
%load('Data/dataset_MNIST.mat')

%% Plot the 9 first images of X_train

figure,
subplot 331, imagesc(X_train(:,:,1)), axis image, axis off, colormap gray
subplot 332, imagesc(X_train(:,:,2)), axis image, axis off, colormap gray
subplot 333, imagesc(X_train(:,:,3)), axis image, axis off, colormap gray
subplot 334, imagesc(X_train(:,:,4)), axis image, axis off, colormap gray
subplot 335, imagesc(X_train(:,:,5)), axis image, axis off, colormap gray
subplot 336, imagesc(X_train(:,:,6)), axis image, axis off, colormap gray
subplot 337, imagesc(X_train(:,:,7)), axis image, axis off, colormap gray
subplot 338, imagesc(X_train(:,:,8)), axis image, axis off, colormap gray
subplot 339, imagesc(X_train(:,:,9)), axis image, axis off, colormap gray

%% Useful dimensions

Nx = size(X_train, 1) ;
Ny = size(X_train, 2) ;
N = Nx*Ny ;
L = size(X_train, 3) ;

%% Parameters for training

% training set as an N x L matrix (each column contains an image)
X_train_mat = reshape(X_train, N, L) ;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO COMPLETE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lipschitz constant of smooth term
beta = (1/L) * norm(X_train_mat, "fro")^2;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% regularization parameters
lambda = 1e-3 ;
delta = 1e-1 ;

% initialization
winit = zeros(N,1) ;

%% Function to check classifier

% binary classifier (returns 1 if answer to question is "yes", and -1 for "no")
d=@(x,w) sign( x'*w ); 
% size of the test dataset
L_test = length(Y_test) ; 
% binary classifier applied to test set
d_test =@(w) d(reshape(X_test, N, L_test), w);

%% Forward-Backward algorithm

Stop_norm = 1e-4 ; 
Stop_crit = 1e-4 ;
ItMax = 10000 ;

[w, perc_error, crit, time] = ...
    FB(winit, X_train_mat, Y_train, lambda, delta, beta, ...
                    d_test, Y_test, ItMax, Stop_norm, Stop_crit) ;

%% Show results

figure, 
subplot 121, plot(cumsum(time), perc_error), xlabel('time (s.)'), ylabel('error (%)'), axis([0 sum(time) 0.1*min(perc_error) 100])
subplot 122, semilogy(cumsum(time), perc_error), xlabel('time (s.)'), ylabel('error (%)'), axis([0 sum(time) 0.1*min(perc_error) 100])

Lab = d_test(w) ;
figure
for i = 1:16
    subplot(4,4,i), imagesc(X_test(:,:,i)), axis image, colormap gray
    xlabel(['Label: true=',num2str(Y_test(i)),' vs. predicted=',num2str(Lab(i))])
end

                
%% Stochastic prox gradient descent algorithm

Stop_norm = 1e-4 ; 
Stop_crit = 1e-4 ;
ItMax = 50000 ;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO COMPLETE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose values for p and theta
p= 0.5;
theta = 1;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('Run stochastic prox gradient descent with')
disp(['p = ',num2str(p)])
disp(['theta = ',num2str(theta)])
[w_S, perc_error_S, crit_S, time_S] = ...
    FB_sto(winit, X_train_mat, Y_train, lambda, delta, p, theta, ...
                    d_test, Y_test, ItMax, Stop_norm, Stop_crit) ;


%% Show results

figure, 
subplot 121, plot(cumsum(time_S), perc_error_S), xlabel('time (s.)'), ylabel('error (%)'), axis([0 sum(time_S) 0.5*min(perc_error_S) 100])
subplot 122, semilogy(cumsum(time_S), perc_error_S), xlabel('time (s.)'), ylabel('error (%)'), axis([0 sum(time_S) 0.5*min(perc_error_S) 100])

Lab_S = d_test(w_S) ;
figure
for i = 1:16
    subplot(4,4,i), imagesc(X_test(:,:,i)), axis image, colormap gray
    xlabel(['Label: true=',num2str(Y_test(i)),' vs. predicted=',num2str(Lab_S(i))])
end


                
%% Stochastic prox gradient descent algorithm with unbiased gradient

Stop_norm = 1e-4 ; 
Stop_crit = 1e-4 ;
ItMax = 50000 ;

p=0.01 ; 

disp('Run stochastic prox gradient descent with unbiased gradient and')
disp(['p = ',num2str(p)])
[w_Sb, perc_error_Sb, crit_Sb, time_Sb] = ...
    FB_sto_unbiased(winit, X_train_mat, Y_train, lambda, delta, p, beta, ...
                    d_test, Y_test, ItMax, Stop_norm, Stop_crit) ;


%% Show results

figure, 
subplot 121, plot(cumsum(time_Sb), perc_error_Sb), xlabel('time (s.)'), ylabel('error (%)'), axis([0 sum(time_Sb) 0.5*min(perc_error_Sb) 100])
subplot 122, semilogy(cumsum(time_Sb), perc_error_Sb), xlabel('time (s.)'), ylabel('error (%)'), axis([0 sum(time_Sb) 0.5*min(perc_error_Sb) 100])


Lab_S = d_test(w_Sb) ;
figure
for i = 1:16
    subplot(4,4,i), imagesc(X_test(:,:,i)), axis image, colormap gray
    xlabel(['Label: true=',num2str(Y_test(i)),' vs. predicted=',num2str(Lab_S(i))])
end