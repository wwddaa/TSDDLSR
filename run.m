%%数据集选取
clear
addpath(['visualization']);
addpath(['algorithm']);
addpath(['tool']);
addpath(['tool/utilities']);
addpath(['tool/PG_Curve-master']);
addpath(['tool/drtoolbox']);
addpath(['Ncut_9']);
addpath(['Functions']);
addpath(['PrecisionRecall']);
 path='datasets/AR_120_26_50_40.mat';
% path='datasets/PoseAll_68_170_32_32.mat';
% path='datasets/Pose29_68_24_64_64.mat';
% path='datasets/Pose09_68_24_64_64.mat';
% path='datasets/ExtYaleB_38_64_96_84.mat';
% path='datasets/ExtYaleB_38_64_48_42.mat';
% path='datasets/FERET_200_7_32_32.mat';
% path='datasets/FERET_200_5_80_80.mat';
% path='datasets/Palm_400_20_48_48.mat';
% path='datasets/Palm_400_20_64_64.mat';
% path='datasets/Palm_100_6_64_64.mat';;
% path='datasets/coil20_20_72_32_32.mat';%OK
% path='datasets/imagenet_10_100_128_384.mat';%ok
% path='datasets/ResNet50imagenet_10_100_49_2048.mat';
% path='datasets/VGG16imagenet_10_100_49_512.mat';
% path='datasets/ResNet50AR_100_26_49_2048.mat';
% path='datasets/ResNet50GT_50_15_49_2048.mat';%OK2
% path='datasets/ResNet50coil20_20_72_49_2048.mat';%?ok2
% path='datasets/ResNet50FERET_200_7_49_2048.mat';%OK
% path='datasets/VGG16AR_100_26_49_512.mat';
% path='datasets/VGG16GT_50_15_49_512.mat';%OK2
% path='datasets/VGG16coil20_20_72_49_512.mat';%ok2
% path='datasets/VGG16FERET_200_7_49_512.mat';%OK
%%
GPU=0;              %是否启用GPU加速
random=1;           %随机选取训练集样本
times=2;            %试验次数（与随机参数配合使用）
train_size=3;      %训练集大小
n_components=300;   %投影维度数
T=5;               %迭代次数
%%
%%单个实验
[output_mean,output_std] = TSDDLSR(path,times,train_size,n_components,T,GPU,random);
fprintf(1,'Accuracy is: %0.2f  %0.2f\n',output_mean,output_std);

