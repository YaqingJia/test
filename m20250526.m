%% 

clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');





%% 
clear all;close all;
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');
sg1 = Conductivity_GT(:,:,80);
mk_wm = abs(sg1 - 0.35) < 1e-4;
mk_gm = abs(sg1 - 0.69) < 1e-4;
mk_csf = abs(sg1 - 2.26) < 1e-4;
c1 = 0.286;
c2 = 1.526*1e-5;
c3 = 11.852;
mk = sg1 ~= 0;
water_content = zeros(size(sg1));
water_content(abs(sg1 - 0.35) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg1 - 0.69) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg1 - 2.26) < 1e-4) = 0.9880;   % 0.9931   2.1440
sg1_w = (c1 + c2 * exp(c3 * water_content)).*mk;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M17.mat');
sg2 = Conductivity_GT(:,:,80);
mk = sg2 ~= 0;
water_content = zeros(size(sg2));
water_content(abs(sg2 - 0.41) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg2 - 0.57) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg2 - 2.09) < 1e-4) = 0.9880;   % 0.9931   2.1440
sg2_w = (c1 + c2 * exp(c3 * water_content)).*mk;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M58.mat');
sg3 = Conductivity_GT(:,:,80);
mk = sg3 ~= 0;
water_content = zeros(size(sg3));
water_content(abs(sg3 - 0.28) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg3 - 0.54) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg3 - 1.99) < 1e-4) = 0.9880;   % 0.9931   2.1440
sg3_w = (c1 + c2 * exp(c3 * water_content)).*mk;

figure;set(gcf,'color','w');
subplot(3,3,1);
imshow(sg1,[0 2.5]);colorbar;colormap(turbo0white);
title('sg1');
subplot(3,3,2);
imshow(sg2,[0 2.5]);colorbar;colormap(turbo0white);
title('sg2');
subplot(3,3,3);
imshow(sg3,[0 2.5]);colorbar;colormap(turbo0white);
title('sg3');
subplot(3,3,4);
imshow(sg1_w,[0 2.5]);colorbar;colormap(turbo0white);
title('sg1\_w');
subplot(3,3,5);
imshow(sg2_w,[0 2.5]);colorbar;colormap(turbo0white);
title('sg2\_w');
subplot(3,3,6);
imshow(sg3_w,[0 2.5]);colorbar;colormap(turbo0white);
title('sg3\_w');
subplot(3,3,7);
imshow(abs(sg1-sg1_w)./(sg1_w+eps),[0 0.2]);colorbar;colormap(turbo0white);
title('RD sg1');
subplot(3,3,8);
imshow(abs(sg2-sg2_w)./(sg2_w+eps),[0 0.2]);colorbar;colormap(turbo0white);
title('RD sg2');
subplot(3,3,9);
imshow(abs(sg3-sg3_w)./(sg3_w+eps),[0 0.2]);colorbar;colormap(turbo0white);
title('RD sg3');

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Tumor\M115.mat');
sg1 = Conductivity_GT(:,:,80);
c1 = 0.286;
c2 = 1.526*1e-5;
c3 = 11.852;
mk = sg1 ~= 0;
water_content = zeros(size(sg1));
water_content(abs(sg1 - 0.34) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg1 - 0.59) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg1 - 2.14) < 1e-4) = 0.9880;   % 0.9931   2.1440
water_content(abs(sg1 - 0.59) < 1e-4) = 0.84;
water_content(abs(sg1 - 1.27) < 1e-4) = 0.84;
sg1_w = (c1 + c2 * exp(c3 * water_content)).*mk;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Tumor\M92.mat');
sg2 = Conductivity_GT(:,:,80);
mk = sg2 ~= 0;
water_content = zeros(size(sg2));
water_content(abs(sg2 - 0.34) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg2 - 0.59) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg2 - 2.14) < 1e-4) = 0.9880;   % 0.9931   2.1440
water_content(abs(sg2 - 0.85) < 1e-4) = 0.84;
water_content(abs(sg2 - 1.03) < 1e-4) = 0.84;
sg2_w = (c1 + c2 * exp(c3 * water_content)).*mk;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Tumor\M102.mat');
sg3 = Conductivity_GT(:,:,80);
mk = sg3 ~= 0;
water_content = zeros(size(sg3));
water_content(abs(sg3 - 0.34) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg3 - 0.59) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg3 - 2.14) < 1e-4) = 0.9880;   % 0.9931   2.1440
water_content(abs(sg3 - 0.71) < 1e-4) = 0.84;
water_content(abs(sg3 - 0.85) < 1e-4) = 0.84;
water_content(abs(sg3 - 1.12) < 1e-4) = 0.84;
sg3_w = (c1 + c2 * exp(c3 * water_content)).*mk;

figure;set(gcf,'color','w');
subplot(3,3,1);
imshow(sg1,[0 2.5]);colorbar;colormap(turbo0white);
title('sg1');
subplot(3,3,2);
imshow(sg2,[0 2.5]);colorbar;colormap(turbo0white);
title('sg2');
subplot(3,3,3);
imshow(sg3,[0 2.5]);colorbar;colormap(turbo0white);
title('sg3');
subplot(3,3,4);
imshow(sg1_w,[0 2.5]);colorbar;colormap(turbo0white);
title('sg1\_w');
subplot(3,3,5);
imshow(sg2_w,[0 2.5]);colorbar;colormap(turbo0white);
title('sg2\_w');
subplot(3,3,6);
imshow(sg3_w,[0 2.5]);colorbar;colormap(turbo0white);
title('sg3\_w');
subplot(3,3,7);
imshow(abs(sg1-sg1_w)./(sg1_w+eps),[0 0.2]);colorbar;colormap(turbo0white);
title('RD sg1');
subplot(3,3,8);
imshow(abs(sg2-sg2_w)./(sg2_w+eps),[0 0.2]);colorbar;colormap(turbo0white);
title('RD sg2');
subplot(3,3,9);
imshow(abs(sg3-sg3_w)./(sg3_w+eps),[0 0.2]);colorbar;colormap(turbo0white);
title('RD sg3');


%% 
% ===================================================================================
% 选出使用无噪声M1训练的最好结果，确定参数，结论：onlyphase34，系数1.3
% ===================================================================================

clear all;close all;

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');
slicestart = 75;
sliceend = 84;
sg = Conductivity_GT(:,:,slicestart:sliceend);
mk = sg ~= 0;
se = strel('cube',13);
se1 = strel('cube',3);
i1 = reshape(mk,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
i = imerode(i1,se,"same");
mk = reshape(i,size(sg));
sg = sg .* mk;

mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;

filepath = 'D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\';
file = dir(filepath);
file = file(3:end);
filenum = length(file);

for i = 1:filenum
    
    filename = file(i).name;
    subfile = dir(fullfile(filepath,filename));
    subpath = fullfile(filepath,filename);
    load(fullfile(subpath,'sg.mat'));
    
    sg_hat = rot90(sg_hat,1);
    sg_hat = medfilt3(sg_hat,[3 3 3]);
    % im = reshape(sg_hat,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
    % sg_hat = reshape(imerode(im,se1,"same"),size(mk));

    Mean.mk_wm.(filename) = mean(sg_hat(mk_wm),'all');
    Mean.mk_gm.(filename) = mean(sg_hat(mk_gm),'all');
    Mean.mk_csf.(filename) = mean(sg_hat(mk_csf),'all');
    Std.mk_wm.(filename) = std(sg_hat(mk_wm),0,'all');
    Std.mk_gm.(filename) = std(sg_hat(mk_gm),0,'all');
    Std.mk_csf.(filename) = std(sg_hat(mk_csf),0,'all');
    RD.mk_wm.(filename) = abs((Mean.mk_wm.(filename) - 0.35)*100./0.35);
    RD.mk_gm.(filename) = abs((Mean.mk_gm.(filename) - 0.69)*100./0.69);
    RD.mk_csf.(filename) = abs((Mean.mk_csf.(filename) - 2.26)*100./2.26);
    % RD.mk_wm.(filename) = mean(abs((sg_hat(mk_wm) - 0.35)*100./0.35),'all');
    % RD.mk_gm.(filename) = mean(abs((sg_hat(mk_gm) - 0.69)*100./0.69),'all');
    % RD.mk_csf.(filename) = mean(abs((sg_hat(mk_csf) - 2.26)*100./2.26),'all');
    RD.sum.(filename) = RD.mk_wm.(filename) + RD.mk_gm.(filename) + RD.mk_csf.(filename);

end



%% 
% ===================================================================================
% 选出不同信噪比水平下，使用M1训练的最好结果，确定参数，结论：6
% ===================================================================================

clear all;close all;

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');
slicestart = 75;
sliceend = 84;
sg = Conductivity_GT(:,:,slicestart:sliceend);
mk = sg ~= 0;
se = strel('cube',13);
se1 = strel('cube',3);
i1 = reshape(mk,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
i = imerode(i1,se,"same");
mk = reshape(i,size(sg));
sg = sg .* mk;

mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;

filepath = 'D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\';
file = dir(filepath);
file = file(3:end);
filenum = length(file);

for i = 1:filenum
    
    filename = file(i).name;
    subfile = dir(fullfile(filepath,filename));
    subpath = fullfile(filepath,filename);
    load(fullfile(subpath,'sg.mat'));
    
    sg_hat = rot90(sg_hat,1);
    sg_hat = medfilt3(sg_hat,[3 3 3]);
    % im = reshape(sg_hat,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
    % sg_hat = reshape(imerode(im,se1,"same"),size(mk));

    Mean.mk_wm.(filename) = mean(sg_hat(mk_wm),'all');
    Mean.mk_gm.(filename) = mean(sg_hat(mk_gm),'all');
    Mean.mk_csf.(filename) = mean(sg_hat(mk_csf),'all');
    Std.mk_wm.(filename) = std(sg_hat(mk_wm),0,'all');
    Std.mk_gm.(filename) = std(sg_hat(mk_gm),0,'all');
    Std.mk_csf.(filename) = std(sg_hat(mk_csf),0,'all');
    RD.mk_wm.(filename) = abs((Mean.mk_wm.(filename) - 0.35)*100./0.35);
    RD.mk_gm.(filename) = abs((Mean.mk_gm.(filename) - 0.69)*100./0.69);
    RD.mk_csf.(filename) = abs((Mean.mk_csf.(filename) - 2.26)*100./2.26);
    RD.sum.(filename) = RD.mk_wm.(filename) + RD.mk_gm.(filename) + RD.mk_csf.(filename);

end



%% 
% ===================================================================================
% 选出使用M89无噪声训练的最好结果，确定参数，结论：onlyphase_M8920
% ===================================================================================

clear all;close all;

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Tumor\M89.mat');
slicestart = 75;
sliceend = 84;
sg = Conductivity_GT(:,:,slicestart:sliceend);
mk = sg ~= 0;
se = strel('cube',13);
i1 = reshape(mk,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
i = imerode(i1,se,"same");
mk = reshape(i,size(sg));
sg = sg .* mk;

mk_wm = abs(sg - 0.34) < 1e-4;
mk_gm = abs(sg - 0.59) < 1e-4;
mk_csf = abs(sg - 2.14) < 1e-4;
mk_tumor = abs(sg - 0.73) < 1e-4;

filepath = 'D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M89\NoiseFree\';
file = dir(filepath);
file = file(3:end);
filenum = length(file);

for i = 1:filenum
    
    filename = file(i).name;
    subfile = dir(fullfile(filepath,filename));
    subpath = fullfile(filepath,filename);
    load(fullfile(subpath,'sg.mat'));
    
    sg_hat = medfilt3(sg_hat,[3 3 3]);
    
    % figure(1);set(gcf,'color','w');
    % subplot(2,3,1);
    % imshow(sg(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('sg');
    % subplot(2,3,2);
    % imshow(sg_hat(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('sg\_hat');
    % subplot(2,3,4);
    % imshow(sg_hat(:,:,2).*mk_wm(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('wm');
    % subplot(2,3,5);
    % imshow(sg_hat(:,:,2).*mk_gm(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('gm');
    % subplot(2,3,6);
    % imshow(sg_hat(:,:,2).*mk_csf(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('csf');

    Mean.mk_wm.(filename) = mean(sg_hat(mk_wm),'all');
    Mean.mk_gm.(filename) = mean(sg_hat(mk_gm),'all');
    Mean.mk_csf.(filename) = mean(sg_hat(mk_csf),'all');
    Mean.mk_tumor.(filename) = mean(sg_hat(mk_tumor),'all');
    Std.mk_wm.(filename) = std(sg_hat(mk_wm),0,'all');
    Std.mk_gm.(filename) = std(sg_hat(mk_gm),0,'all');
    Std.mk_csf.(filename) = std(sg_hat(mk_csf),0,'all');
    Std.mk_tumor.(filename) = std(sg_hat(mk_tumor),0,'all');
    RD.mk_wm.(filename) = abs((Mean.mk_wm.(filename) - 0.34)*100./0.34);
    RD.mk_gm.(filename) = abs((Mean.mk_gm.(filename) - 0.59)*100./0.59);
    RD.mk_csf.(filename) = abs((Mean.mk_csf.(filename) - 2.14)*100./2.14);
    RD.mk_tumor.(filename) = abs((Mean.mk_tumor.(filename) - 0.73)*100./0.73);
    RD.sum.(filename) = RD.mk_wm.(filename) + RD.mk_gm.(filename) + RD.mk_csf.(filename) + RD.mk_tumor.(filename);

end

%% 
% ===================================================================================
% 选出cr-EPT的最好结果，确定参数，结论：1e-9
% ===================================================================================

clear all;close all;

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');
slicestart = 75;
sliceend = 84;
sg = Conductivity_GT(:,:,slicestart:sliceend);
mk = sg ~= 0;
se = strel('cube',13);
i1 = reshape(mk,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
i = imerode(i1,se,"same");
mk = reshape(i,size(sg));
sg = sg .* mk;

mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;

filepath = 'D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\crEPT\';
file = dir(filepath);
file = file(4:end);
filenum = length(file);

for i = 1:filenum
    
    filename = file(i).name;
    filename_ = strsplit(filename,'.');
    subfile = dir(fullfile(filepath,filename));
    subpath = fullfile(filepath,filename);

    sg_cr = h5read(subpath,'/sigma');
    sg_cr = rot90(sg_cr,1);
    sg_cr = sg_cr(:,:,3:12);
    sg_cr(isnan(sg_cr)) = 0;
    sg_cr = sg_cr .* mk;
    sg_cr = medfilt3(sg_cr,[3 3 3]);
    
    % figure(1);set(gcf,'color','w');
    % subplot(2,3,1);
    % imshow(sg(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('sg');
    % subplot(2,3,2);
    % imshow(sg_hat(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('sg\_hat');
    % subplot(2,3,4);
    % imshow(sg_hat(:,:,2).*mk_wm(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('wm');
    % subplot(2,3,5);
    % imshow(sg_hat(:,:,2).*mk_gm(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('gm');
    % subplot(2,3,6);
    % imshow(sg_hat(:,:,2).*mk_csf(:,:,2),[0 3]);colorbar;colormap turbo;
    % title('csf');

    Mean.mk_wm.(filename_{1}) = mean(sg_cr(mk_wm),'all');
    Mean.mk_gm.(filename_{1}) = mean(sg_cr(mk_gm),'all');
    Mean.mk_csf.(filename_{1}) = mean(sg_cr(mk_csf),'all');
    Std.mk_wm.(filename_{1}) = std(sg_cr(mk_wm),0,'all');
    Std.mk_gm.(filename_{1}) = std(sg_cr(mk_gm),0,'all');
    Std.mk_csf.(filename_{1}) = std(sg_cr(mk_csf),0,'all');
    RD.mk_wm.(filename_{1}) = abs((Mean.mk_wm.(filename_{1}) - 0.35)*100./0.35);
    RD.mk_gm.(filename_{1}) = abs((Mean.mk_gm.(filename_{1}) - 0.69)*100./0.69);
    RD.mk_csf.(filename_{1}) = abs((Mean.mk_csf.(filename_{1}) - 2.26)*100./2.26);
    RD.sum.(filename_{1}) = RD.mk_wm.(filename_{1}) + RD.mk_gm.(filename_{1}) + RD.mk_csf.(filename_{1});

end



%% 
clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\PINN_onlyphase34\sg.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\pinn_onlyphase.mat');
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');

slicestart = 75;
sliceend = 84;
ystart = 40;
yend = 209;
xstart = 65;
xend = 194;
slice1 = 2;
slice2 = 5;
slice3 = 8;
snr = 84;
cv = 0.16;
w = 2*pi*128e6;
u0 = 4e-7*pi;
A = 0.89;
B = 0.5;
TR1 = 700*1e-3;
TR2 = 3000*1e-3;
alpha = pi/2;
C = 1;
c1 = 0.286;
c2 = 1.526*1e-5;
c3 = 11.852;

l_phase = gxx_phase(:,:,slicestart:sliceend) + gyy_phase(:,:,slicestart:sliceend) + gzz_phase(:,:,slicestart:sliceend);
l_phase = rot90(l_phase,3);
sg_h = l_phase/(2*w*u0);

sg = conductivity(:,:,slicestart:sliceend);
sg = rot90(sg,3);
water_content = zeros(size(sg));
water_content(abs(sg - 0.35) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg - 0.69) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg - 2.26) < 1e-4) = 0.9880;   % 0.9931   2.1440
mk = water_content ~=0;
% sg2 = c1 + c2 * exp(c3 * water_content);
% sg2 = sg2.*mk;

T1 = B./(1./water_content - A);
T1(isnan(T1)) = 0;
SI1 = C*sin(alpha) * (1 - exp(-TR1./T1)) ./ (1 + cos(alpha)*exp(-TR1./T1));
SI1(SI1 == 1) = 0;
SI2 = C*sin(alpha) * (1 - exp(-TR2./T1)) ./ (1 + cos(alpha)*exp(-TR2./T1));
SI2(SI2 == 1) = 0;

SI1 = SI1 + SI1 / snr .*randn(size(SI1));
SI2 = SI2 + SI2 / snr .*randn(size(SI2));
T11 = -TR1 ./ (log(1-SI1));
T12 = -TR2 ./ (log(abs(1-SI2)));

water_content = (1./(A + B./T12)).*mk;
water_content(water_content >= 1) = 1;
corr(water_content(water_content ~= 0),sg( sg~=0 ))
% water_content = water_content + water_content / snr .* randn(size(water_content));

% figure(1);set(gcf,'Color','w');
% subplot(1,3,1);
% imshow(T11(:,:,5),[0 5]);colorbar;colormap turbo;
% subplot(1,3,2);
% imshow(T12(:,:,5),[0 5]);colorbar;colormap turbo;
% subplot(1,3,3);
% imshow(T11(:,:,5)-T12(:,:,5),[]);colorbar;colormap turbo;

sg1 = c1 + c2 * exp(c3 * water_content);
sg1(abs(sg1 - c1) < 1e-4) = 0;
mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;
sg1_wm = sg1 .* mk_wm;
sg1_gm = sg1 .* mk_gm;
sg1_csf = sg1 .* mk_csf;
Mean.sg1_wm = mean(sg1_wm(sg1_wm~=0));
Mean.sg1_gm = mean(sg1_gm(sg1_gm~=0));
Mean.sg1_csf = mean(sg1_csf(sg1_csf~=0));

sg2 = zeros(size(sg1));
sg2(sg1_wm~=0) = random('Normal',Mean.sg1_wm,Mean.sg1_wm*cv,size(sg1_wm(sg1_wm~=0)));
sg2(sg1_gm~=0) = random('Normal',Mean.sg1_gm,Mean.sg1_gm*cv,size(sg1_gm(sg1_gm~=0)));
sg2(sg1_csf~=0) = random('Normal',Mean.sg1_csf,Mean.sg1_csf*cv,size(sg1_csf(sg1_csf~=0)));

% sg_hat = rot90(medfilt3(sg_hat,[3 3 3]),3);
% se1 = strel('cube',3);
% im = reshape(sg_hat,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
% sg_hat_pw = reshape(imerode(im,se1,"same"),size(mk));
sg_hat_pw = rot90(medfilt3(sg_hat,[3 3 3]),3);
sg_cr = h5read('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\crEPT\output_coe_1e_09.h5','/sigma');
sg_cr = rot90(sg_cr,3);

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\PINN_onlyphase0\sg.mat');
sg_hat_p = rot90(sg_hat,3);

figure(1);set(gcf,'Color','w');
subplot(6,3,1);
imshow(sg(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('GT');
subplot(6,3,4);
imshow(sg_h(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('H-EPT');
subplot(6,3,7);
imshow(sg_cr(:,:,slice1+3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('cr-EPT');
subplot(6,3,10);
imshow(sg2(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('w-EPT');
subplot(6,3,13);
imshow(sg_hat_p(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('only PINN');
subplot(6,3,16);
imshow(sg_hat_pw(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wPINN');

subplot(6,3,2);
imshow(sg(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('GT');
subplot(6,3,5);
imshow(sg_h(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('H-EPT');
subplot(6,3,8);
imshow(sg_cr(:,:,slice2+3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('cr-EPT');
subplot(6,3,11);
imshow(sg2(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('w-EPT');
subplot(6,3,14);
imshow(sg_hat_p(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('only PINN');
subplot(6,3,17);
imshow(sg_hat_pw(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wPINN');

subplot(6,3,3);
imshow(sg(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('GT');
subplot(6,3,6);
imshow(sg_h(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('H-EPT');
subplot(6,3,9);
imshow(sg_cr(:,:,slice3+3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('cr-EPT');
subplot(6,3,12);
imshow(sg2(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('w-EPT');
subplot(6,3,15);
imshow(sg_hat_p(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('only PINN');
subplot(6,3,18);
imshow(sg_hat_pw(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wPINN');



figure(2);set(gcf,'Color','w');
subplot(2,3,1);
imshow(sg(:,:,slice2).*mk_wm(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title('wm');
subplot(2,3,2);
imshow(sg(:,:,slice2).*mk_gm(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title('gm');
subplot(2,3,3);
imshow(sg(:,:,slice2).*mk_csf(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title('csf');
subplot(2,3,4);
imshow(sg_hat_pw(:,:,slice2).*mk_wm(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title('wm');
subplot(2,3,5);
imshow(sg_hat_pw(:,:,slice2).*mk_gm(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title('gm');
subplot(2,3,6);
imshow(sg_hat_pw(:,:,slice2).*mk_csf(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title('csf');



xslice = 135;
sg_ = sg(xslice,:,slice1);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_h_ = sg_h(xslice,:,slice1);
sg_h_ = sg_h_(mk_);
sg_cr_ = sg_cr(xslice,:,slice1+3);
sg_cr_ = sg_cr_(mk_);
sg2_ = sg2(xslice,:,slice1);
sg2_ = sg2_(mk_);
sg_hat_p_ = sg_hat_p(xslice,:,slice1);
sg_hat_p_ = sg_hat_p_(mk_);
sg_hat_pw_ = sg_hat_pw(xslice,:,slice1);
sg_hat_pw_ = sg_hat_pw_(mk_);

figure('Name','profile');
pt = 1:length(sg_);
subplot(2,3,1);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_h_,'m--.',pt,sg_hat_pw_,'r--.',pt,sg2_,'b--.');
hold on;
plot(pt,sg_hat_p_,'--.','Color','#0072BD');
hold on;
plot(pt,sg_cr_,'--.','Color','#77AC30');
ylim(subplot(2,3,1),[0,3]);
% ylim([0 3]);
legend('GT','H-EPT','Wc-PINN','wEPT','only PDE','cr-EPT','location','northeast','Orientation','horizontal','NumColumns',3);
legend('boxoff');
title(['sg slice',num2str(slice1)]);
hold off;

sg_ = sg(xslice,:,slice2);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_h_ = sg_h(xslice,:,slice2);
sg_h_ = sg_h_(mk_);
sg_cr_ = sg_cr(xslice,:,slice2+3);
sg_cr_ = sg_cr_(mk_);
sg2_ = sg2(xslice,:,slice2);
sg2_ = sg2_(mk_);
sg_hat_p_ = sg_hat_p(xslice,:,slice2);
sg_hat_p_ = sg_hat_p_(mk_);
sg_hat_pw_ = sg_hat_pw(xslice,:,slice2);
sg_hat_pw_ = sg_hat_pw_(mk_);

pt = 1:length(sg_);
subplot(2,3,2);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_h_,'m--.',pt,sg_hat_pw_,'r--.',pt,sg2_,'b--.');
hold on;
plot(pt,sg_hat_p_,'--.','Color','#0072BD');
hold on;
plot(pt,sg_cr_,'--.','Color','#77AC30');
ylim(subplot(2,3,2),[0,3]);
legend('GT','H-EPT','Wc-PINN','wEPT','only PDE','cr-EPT','location','northeast','Orientation','horizontal','NumColumns',3);
legend('boxoff');
title(['sg slice',num2str(slice2)]);
hold off;

sg_ = sg(xslice,:,slice3);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_h_ = sg_h(xslice,:,slice3);
sg_h_ = sg_h_(mk_);
sg_cr_ = sg_cr(xslice,:,slice3+3);
sg_cr_ = sg_cr_(mk_);
sg2_ = sg2(xslice,:,slice3);
sg2_ = sg2_(mk_);
sg_hat_p_ = sg_hat_p(xslice,:,slice3);
sg_hat_p_ = sg_hat_p_(mk_);
sg_hat_pw_ = sg_hat_pw(xslice,:,slice3);
sg_hat_pw_ = sg_hat_pw_(mk_);

pt = 1:length(sg_);
subplot(2,3,3);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_h_,'m--.',pt,sg_hat_pw_,'r--.',pt,sg2_,'b--.');
hold on;
plot(pt,sg_hat_p_,'--.','Color','#0072BD');
hold on;
plot(pt,sg_cr_,'--.','Color','#77AC30');
ylim(subplot(2,3,3),[0,3]);
legend('GT','H-EPT','Wc-PINN','wEPT','only PDE','cr-EPT','location','northeast','Orientation','horizontal','NumColumns',3);
legend('boxoff');
title(['sg slice',num2str(slice3)]);
hold off;





%% 
clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M89\NoiseFree\PINN_onlyphase_M8924\sg.mat');
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Tumor\M89.mat');

slicestart = 75;
sliceend = 84;
ystart = 40;
yend = 209;
xstart = 65;
xend = 194;
slice1 = 2;
slice2 = 5;
slice3 = 8;
snr = 84;
cv = 0.16;
w = 2*pi*128e6;
u0 = 4e-7*pi;
A = 0.89;
B = 0.5;
TR1 = 700*1e-3;
TR2 = 3000*1e-3;
alpha = pi/2;
C = 1;
c1 = 0.286;
c2 = 1.526*1e-5;
c3 = 11.852;

sg = Conductivity_GT(:,:,slicestart:sliceend);
mk = sg ~= 0;
se = strel('cube',13);
i1 = reshape(mk,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
i = imerode(i1,se,"same");
mk = reshape(i,size(sg));
sg = sg .* mk;

water_content = zeros(size(sg));
water_content(abs(sg - 0.34) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg - 0.59) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg - 2.14) < 1e-4) = 0.9880;   % 0.9931   2.1440
water_content(abs(sg - 0.73) < 1e-4) = 0.84;
mk = water_content ~=0;
% sg2 = c1 + c2 * exp(c3 * water_content);
% sg2 = sg2.*mk;

T1 = B./(1./water_content - A);
T1(isnan(T1)) = 0;
SI1 = C*sin(alpha) * (1 - exp(-TR1./T1)) ./ (1 + cos(alpha)*exp(-TR1./T1));
SI1(SI1 == 1) = 0;
SI2 = C*sin(alpha) * (1 - exp(-TR2./T1)) ./ (1 + cos(alpha)*exp(-TR2./T1));
SI2(SI2 == 1) = 0;

SI1 = SI1 + SI1 / snr .*randn(size(SI1));
SI2 = SI2 + SI2 / snr .*randn(size(SI2));
T11 = -TR1 ./ (log(1-SI1));
T12 = -TR2 ./ (log(abs(1-SI2)));

water_content = (1./(A + B./T12)).*mk;
water_content(water_content >= 1) = 1;
corr(water_content(water_content ~= 0),sg( sg~=0 ))
% water_content = water_content + water_content / snr .* randn(size(water_content));

% figure(1);set(gcf,'Color','w');
% subplot(1,3,1);
% imshow(T11(:,:,5),[0 5]);colorbar;colormap turbo;
% subplot(1,3,2);
% imshow(T12(:,:,5),[0 5]);colorbar;colormap turbo;
% subplot(1,3,3);
% imshow(T11(:,:,5)-T12(:,:,5),[]);colorbar;colormap turbo;

sg1 = c1 + c2 * exp(c3 * water_content);
sg1(abs(sg1 - c1) < 1e-4) = 0;
mk_wm = abs(sg - 0.34) < 1e-4;
mk_gm = abs(sg - 0.59) < 1e-4;
mk_csf = abs(sg - 2.14) < 1e-4;
mk_tumor = abs(sg - 0.73) < 1e-4;
sg1_wm = sg1 .* mk_wm;
sg1_gm = sg1 .* mk_gm;
sg1_csf = sg1 .* mk_csf;
sg1_tumor = sg1 .* mk_tumor;
Mean.sg1_wm = mean(sg1_wm(sg1_wm~=0));
Mean.sg1_gm = mean(sg1_gm(sg1_gm~=0));
Mean.sg1_csf = mean(sg1_csf(sg1_csf~=0));
Mean.sg1_tumor = mean(sg1_tumor(sg1_tumor~=0));

sg2 = zeros(size(sg1));
sg2(sg1_wm~=0) = random('Normal',Mean.sg1_wm,Mean.sg1_wm*cv,size(sg1_wm(sg1_wm~=0)));
sg2(sg1_gm~=0) = random('Normal',Mean.sg1_gm,Mean.sg1_gm*cv,size(sg1_gm(sg1_gm~=0)));
sg2(sg1_csf~=0) = random('Normal',Mean.sg1_csf,Mean.sg1_csf*cv,size(sg1_csf(sg1_csf~=0)));
sg2(sg1_tumor~=0) = random('Normal',Mean.sg1_tumor,Mean.sg1_tumor*cv,size(sg1_tumor(sg1_tumor~=0)));

sg_hat_pw = medfilt3(sg_hat,[3 3 3]);

Mean.mk_wm.sg_hat_pw = mean(sg_hat_pw(mk_wm),'all');
Mean.mk_gm.sg_hat_pw = mean(sg_hat_pw(mk_gm),'all');
Mean.mk_csf.sg_hat_pw = mean(sg_hat_pw(mk_csf),'all');
Mean.mk_tumor.sg_hat_pw = mean(sg_hat_pw(mk_tumor),'all');

Std.mk_wm.sg_hat_pw = std(sg_hat_pw(mk_wm),0,'all');
Std.mk_gm.sg_hat_pw = std(sg_hat_pw(mk_gm),0,'all');
Std.mk_csf.sg_hat_pw = std(sg_hat_pw(mk_csf),0,'all');
Std.mk_tumor.sg_hat_pw = std(sg_hat_pw(mk_tumor),0,'all');

figure(1);set(gcf,'Color','w');
subplot(2,3,1);
imshow(sg(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('GT');
subplot(2,3,4);
imshow(sg_hat_pw(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wPINN');
subplot(2,3,2);
imshow(sg(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('GT');
subplot(2,3,5);
imshow(sg_hat_pw(:,:,slice2),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wPINN');
subplot(2,3,3);
imshow(sg(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('GT');
subplot(2,3,6);
imshow(sg_hat_pw(:,:,slice3),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wPINN');

figure(2);set(gcf,'Color','w');
subplot(2,4,1);
imshow(sg(:,:,slice1).*mk_wm(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wm GT');
subplot(2,4,5);
imshow(sg_hat_pw(:,:,slice1).*mk_wm(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('wm wPINN');
subplot(2,4,2);
imshow(sg(:,:,slice1).*mk_gm(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('gm GT');
subplot(2,4,6);
imshow(sg_hat_pw(:,:,slice1).*mk_gm(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('gm wPINN');
subplot(2,4,3);
imshow(sg(:,:,slice1).*mk_csf(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('csf GT');
subplot(2,4,7);
imshow(sg_hat_pw(:,:,slice1).*mk_csf(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('csf wPINN');
subplot(2,4,4);
imshow(sg(:,:,slice1).*mk_tumor(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('tumor GT');
subplot(2,4,8);
imshow(sg_hat_pw(:,:,slice1).*mk_tumor(:,:,slice1),[0 3],'InitialMagnification','fit');colorbar;colormap(turbo0white);
title('tumor wPINN');

xslice = 152;
sg_ = sg(xslice,:,slice1);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_hat_pw_ = sg_hat_pw(xslice,:,slice1);
sg_hat_pw_ = sg_hat_pw_(mk_);

figure('Name','profile');
pt = 1:length(sg_);
subplot(2,3,1);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_hat_pw_,'r--.');
ylim(subplot(2,3,1),[0,3]);
xlim(subplot(2,3,1),[0,125]);
% ylim([0 3]);
legend('GT','Wc-PINN','location','northeast','Orientation','horizontal','NumColumns',3);
legend('boxoff');
title(['sg slice',num2str(slice1)]);

sg_ = sg(xslice,:,slice2);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_hat_pw_ = sg_hat_pw(xslice,:,slice2);
sg_hat_pw_ = sg_hat_pw_(mk_);

pt = 1:length(sg_);
subplot(2,3,2);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_hat_pw_,'r--.');
ylim(subplot(2,3,2),[0,3]);
xlim(subplot(2,3,2),[0,125]);
legend('GT','Wc-PINN','location','northeast','Orientation','horizontal','NumColumns',3);
legend('boxoff');
title(['sg slice',num2str(slice2)]);
hold off;

sg_ = sg(xslice,:,slice3);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_hat_pw_ = sg_hat_pw(xslice,:,slice3);
sg_hat_pw_ = sg_hat_pw_(mk_);

pt = 1:length(sg_);
subplot(2,3,3);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_hat_pw_,'r--.');
ylim(subplot(2,3,3),[0,3]);
xlim(subplot(2,3,3),[0,125]);
legend('GT','Wc-PINN','location','northeast','Orientation','horizontal','NumColumns',3);
legend('boxoff');
title(['sg slice',num2str(slice3)]);



%% 
clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\PINN_onlyphase30db6\sg.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\pinn_onlyphase.mat');
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');

slicestart = 75;
sliceend = 84;
ystart = 40;
yend = 209;
xstart = 65;
xend = 194;
slice1 = 2;
slice2 = 5;
slice3 = 8;

sg = conductivity(:,:,slicestart:sliceend);
sg = rot90(sg,3);
mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;

sg_hat_pw30db = rot90(medfilt3(sg_hat,[3 3 3]),2);
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\PINN_onlyphase50db6\sg.mat');
sg_hat_pw50db = rot90(medfilt3(sg_hat,[3 3 3]),2);
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\PINN_onlyphase70db6\sg.mat');
sg_hat_pw70db = rot90(medfilt3(sg_hat,[3 3 3]),2);

figure;set(gcf,'color','w');
subplot(4,3,1);
imshow(sg(:,:,slice1),[0 3]);colorbar;
title(['GT slice' num2str(slice1)]);
subplot(4,3,4);
imshow(sg_hat_pw30db(:,:,slice1),[0 3]);colorbar;
title(['30db slice' num2str(slice1)]);
subplot(4,3,7);
imshow(sg_hat_pw50db(:,:,slice1),[0 3]);colorbar;
title(['50db slice' num2str(slice1)]);
subplot(4,3,10);
imshow(sg_hat_pw70db(:,:,slice1),[0 3]);colorbar;
title(['70db slice' num2str(slice1)]);

subplot(4,3,2);
imshow(sg(:,:,slice2),[0 3]);colorbar;
title(['GT slice' num2str(slice2)]);
subplot(4,3,5);
imshow(sg_hat_pw30db(:,:,slice2),[0 3]);colorbar;
title(['30db slice' num2str(slice2)]);
subplot(4,3,8);
imshow(sg_hat_pw50db(:,:,slice2),[0 3]);colorbar;
title(['50db slice' num2str(slice2)]);
subplot(4,3,11);
imshow(sg_hat_pw70db(:,:,slice2),[0 3]);colorbar;
title(['70db slice' num2str(slice2)]);

subplot(4,3,3);
imshow(sg(:,:,slice3),[0 3]);colorbar;
title(['GT slice' num2str(slice3)]);
subplot(4,3,6);
imshow(sg_hat_pw30db(:,:,slice3),[0 3]);colorbar;
title(['30db slice' num2str(slice3)]);
subplot(4,3,9);
imshow(sg_hat_pw50db(:,:,slice3),[0 3]);colorbar;
title(['50db slice' num2str(slice3)]);
subplot(4,3,12);
imshow(sg_hat_pw70db(:,:,slice3),[0 3]);colorbar;colormap(turbo0white);
title(['70db slice' num2str(slice3)]);



figure;set(gcf,'color','w');
subplot(4,3,1);
imshow(sg(:,:,slice2).*mk_wm(:,:,slice2),[0 3]);colorbar;
title(['GT WM slice' num2str(slice2)]);
subplot(4,3,4);
imshow(sg_hat_pw30db(:,:,slice2).*mk_wm(:,:,slice2),[0 3]);colorbar;
title(['30db WM slice' num2str(slice2)]);
subplot(4,3,7);
imshow(sg_hat_pw50db(:,:,slice2).*mk_wm(:,:,slice2),[0 3]);colorbar;
title(['50db WM slice' num2str(slice2)]);
subplot(4,3,10);
imshow(sg_hat_pw70db(:,:,slice2).*mk_wm(:,:,slice2),[0 3]);colorbar;
title(['70db WM slice' num2str(slice2)]);

subplot(4,3,2);
imshow(sg(:,:,slice2).*mk_gm(:,:,slice2),[0 3]);colorbar;
title(['GT GM slice' num2str(slice2)]);
subplot(4,3,5);
imshow(sg_hat_pw30db(:,:,slice2).*mk_gm(:,:,slice2),[0 3]);colorbar;
title(['30db GM slice' num2str(slice2)]);
subplot(4,3,8);
imshow(sg_hat_pw50db(:,:,slice2).*mk_gm(:,:,slice2),[0 3]);colorbar;
title(['50db GM slice' num2str(slice2)]);
subplot(4,3,11);
imshow(sg_hat_pw70db(:,:,slice2).*mk_gm(:,:,slice2),[0 3]);colorbar;
title(['70db GM slice' num2str(slice2)]);

subplot(4,3,3);
imshow(sg(:,:,slice2).*mk_csf(:,:,slice2),[0 3]);colorbar;
title(['GT CSF slice' num2str(slice2)]);
subplot(4,3,6);
imshow(sg_hat_pw30db(:,:,slice2).*mk_csf(:,:,slice2),[0 3]);colorbar;
title(['30db CSF slice' num2str(slice2)]);
subplot(4,3,9);
imshow(sg_hat_pw50db(:,:,slice2).*mk_csf(:,:,slice2),[0 3]);colorbar;
title(['50db CSF slice' num2str(slice2)]);
subplot(4,3,12);
imshow(sg_hat_pw70db(:,:,slice2).*mk_csf(:,:,slice2),[0 3]);colorbar;colormap(turbo0white);
title(['70db CSF slice' num2str(slice2)]);


xslice = 135;
sg_ = sg(xslice,:,slice1);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_hat_pw30db_ = sg_hat_pw30db(xslice,:,slice1);
sg_hat_pw30db_ = sg_hat_pw30db_(mk_);
sg_hat_pw50db_ = sg_hat_pw50db(xslice,:,slice1);
sg_hat_pw50db_ = sg_hat_pw50db_(mk_);
sg_hat_pw70db_ = sg_hat_pw70db(xslice,:,slice1);
sg_hat_pw70db_ = sg_hat_pw70db_(mk_);

figure('Name','profile');
pt = 1:length(sg_);
subplot(2,3,1);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_hat_pw30db_,'m--.',pt,sg_hat_pw50db_,'r--.',pt,sg_hat_pw70db_,'b--.');
ylim(subplot(2,3,1),[0,3]);
% ylim([0 3]);
legend('GT','30db','50db','70db','location','northeast','Orientation','horizontal','NumColumns',2);
legend('boxoff');
title(['sg slice',num2str(slice1)]);

sg_ = sg(xslice,:,slice2);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_hat_pw30db_ = sg_hat_pw30db(xslice,:,slice2);
sg_hat_pw30db_ = sg_hat_pw30db_(mk_);
sg_hat_pw50db_ = sg_hat_pw50db(xslice,:,slice2);
sg_hat_pw50db_ = sg_hat_pw50db_(mk_);
sg_hat_pw70db_ = sg_hat_pw70db(xslice,:,slice2);
sg_hat_pw70db_ = sg_hat_pw70db_(mk_);

pt = 1:length(sg_);
subplot(2,3,2);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_hat_pw30db_,'m--.',pt,sg_hat_pw50db_,'r--.',pt,sg_hat_pw70db_,'b--.');
ylim(subplot(2,3,2),[0,3]);
% ylim([0 3]);
legend('GT','30db','50db','70db','location','northeast','Orientation','horizontal','NumColumns',2);
legend('boxoff');
title(['sg slice',num2str(slice2)]);

sg_ = sg(xslice,:,slice3);
mk_ = sg_~=0;
sg_ = sg_(mk_);
sg_hat_pw30db_ = sg_hat_pw30db(xslice,:,slice3);
sg_hat_pw30db_ = sg_hat_pw30db_(mk_);
sg_hat_pw50db_ = sg_hat_pw50db(xslice,:,slice3);
sg_hat_pw50db_ = sg_hat_pw50db_(mk_);
sg_hat_pw70db_ = sg_hat_pw70db(xslice,:,slice3);
sg_hat_pw70db_ = sg_hat_pw70db_(mk_);

pt = 1:length(sg_);
subplot(2,3,3);set(gcf,'Color','w');
plot(pt,sg_,'k-',pt,sg_hat_pw30db_,'m--.',pt,sg_hat_pw50db_,'r--.',pt,sg_hat_pw70db_,'b--.');
ylim(subplot(2,3,3),[0,3]);
% ylim([0 3]);
legend('GT','30db','50db','70db','location','northeast','Orientation','horizontal','NumColumns',2);
legend('boxoff');
title(['sg slice',num2str(slice2)]);


%% 
clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\PINN_onlyphase34\sg.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\pinn_onlyphase.mat');
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');

slicestart = 75;
sliceend = 84;
ystart = 40;
yend = 209;
xstart = 65;
xend = 194;
slice1 = 2;
slice2 = 5;
slice3 = 8;
snr = 84;
cv = 0.16;
w = 2*pi*128e6;
u0 = 4e-7*pi;
A = 0.89;
B = 0.5;
TR1 = 700*1e-3;
TR2 = 3000*1e-3;
alpha = pi/2;
C = 1;
c1 = 0.286;
c2 = 1.526*1e-5;
c3 = 11.852;

l_phase = gxx_phase(:,:,slicestart:sliceend) + gyy_phase(:,:,slicestart:sliceend) + gzz_phase(:,:,slicestart:sliceend);
l_phase = rot90(l_phase,3);
sg_h = l_phase/(2*w*u0);

sg = conductivity(:,:,slicestart:sliceend);
sg = rot90(sg,3);
water_content = zeros(size(sg));
water_content(abs(sg - 0.35) < 1e-4) = 0.6957;   % 0.7038   0.3441
water_content(abs(sg - 0.69) < 1e-4) = 0.8341;   % 0.8593   0.5858
water_content(abs(sg - 2.26) < 1e-4) = 0.9880;   % 0.9931   2.1440
mk = water_content ~=0;
% sg2 = c1 + c2 * exp(c3 * water_content);
% sg2 = sg2.*mk;

T1 = B./(1./water_content - A);
T1(isnan(T1)) = 0;
SI1 = C*sin(alpha) * (1 - exp(-TR1./T1)) ./ (1 + cos(alpha)*exp(-TR1./T1));
SI1(SI1 == 1) = 0;
SI2 = C*sin(alpha) * (1 - exp(-TR2./T1)) ./ (1 + cos(alpha)*exp(-TR2./T1));
SI2(SI2 == 1) = 0;

SI1 = SI1 + SI1 / snr .*randn(size(SI1));
SI2 = SI2 + SI2 / snr .*randn(size(SI2));
T11 = -TR1 ./ (log(1-SI1));
T12 = -TR2 ./ (log(abs(1-SI2)));

water_content = (1./(A + B./T12)).*mk;
water_content(water_content >= 1) = 1;
corr(water_content(water_content ~= 0),sg( sg~=0 ))
% water_content = water_content + water_content / snr .* randn(size(water_content));

% figure(1);set(gcf,'Color','w');
% subplot(1,3,1);
% imshow(T11(:,:,5),[0 5]);colorbar;colormap turbo;
% subplot(1,3,2);
% imshow(T12(:,:,5),[0 5]);colorbar;colormap turbo;
% subplot(1,3,3);
% imshow(T11(:,:,5)-T12(:,:,5),[]);colorbar;colormap turbo;

sg1 = c1 + c2 * exp(c3 * water_content);
sg1(abs(sg1 - c1) < 1e-4) = 0;
mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;
sg1_wm = sg1 .* mk_wm;
sg1_gm = sg1 .* mk_gm;
sg1_csf = sg1 .* mk_csf;
Mean.sg1_wm = mean(sg1_wm(sg1_wm~=0));
Mean.sg1_gm = mean(sg1_gm(sg1_gm~=0));
Mean.sg1_csf = mean(sg1_csf(sg1_csf~=0));

sg2 = zeros(size(sg1));
sg2(sg1_wm~=0) = random('Normal',Mean.sg1_wm,Mean.sg1_wm*cv,size(sg1_wm(sg1_wm~=0)));
sg2(sg1_gm~=0) = random('Normal',Mean.sg1_gm,Mean.sg1_gm*cv,size(sg1_gm(sg1_gm~=0)));
sg2(sg1_csf~=0) = random('Normal',Mean.sg1_csf,Mean.sg1_csf*cv,size(sg1_csf(sg1_csf~=0)));

sg_cr = h5read('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\crEPT\output_coe_1e_09.h5','/sigma');
sg_cr = rot90(sg_cr(:,:,4:end-3),3);

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\PINN_onlyphase34\sg.mat');
sg_hat_pw = rot90(medfilt3(sg_hat,[3 3 3]),3);
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\PINN_onlyphase70db6\sg.mat');
sg_hat_pw70db = rot90(medfilt3(sg_hat,[3 3 3]),2);
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\PINN_onlyphase50db6\sg.mat');
sg_hat_pw50db = rot90(medfilt3(sg_hat,[3 3 3]),2);
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\Noisy\PINN_onlyphase30db6\sg.mat');
sg_hat_pw30db = rot90(medfilt3(sg_hat,[3 3 3]),2);
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\PINN_onlyphase0\sg.mat');
sg_hat_p = rot90(sg_hat,3);

Mean.mk_wm.sg_h = mean(sg_h(mk_wm),'all');
Mean.mk_gm.sg_h = mean(sg_h(mk_gm),'all');
Mean.mk_csf.sg_h = mean(sg_h(mk_csf),'all');
Mean.mk_wm.sg_cr = mean(sg_cr(mk_wm),'all');
Mean.mk_gm.sg_cr = mean(sg_cr(mk_gm),'all');
Mean.mk_csf.sg_cr = mean(sg_cr(mk_csf),'all');
Mean.mk_wm.sg2 = mean(sg2(mk_wm),'all');
Mean.mk_gm.sg2 = mean(sg2(mk_gm),'all');
Mean.mk_csf.sg2 = mean(sg2(mk_csf),'all');
Mean.mk_wm.sg_hat_p = mean(sg_hat_p(mk_wm),'all');
Mean.mk_gm.sg_hat_p = mean(sg_hat_p(mk_gm),'all');
Mean.mk_csf.sg_hat_p = mean(sg_hat_p(mk_csf),'all');
Mean.mk_wm.sg_hat_pw = mean(sg_hat_pw(mk_wm),'all');
Mean.mk_gm.sg_hat_pw = mean(sg_hat_pw(mk_gm),'all');
Mean.mk_csf.sg_hat_pw = mean(sg_hat_pw(mk_csf),'all');
Mean.mk_wm.sg_hat_pw70db = mean(sg_hat_pw70db(mk_wm),'all');
Mean.mk_gm.sg_hat_pw70db = mean(sg_hat_pw70db(mk_gm),'all');
Mean.mk_csf.sg_hat_pw70db = mean(sg_hat_pw70db(mk_csf),'all');
Mean.mk_wm.sg_hat_pw50db = mean(sg_hat_pw50db(mk_wm),'all');
Mean.mk_gm.sg_hat_pw50db = mean(sg_hat_pw50db(mk_gm),'all');
Mean.mk_csf.sg_hat_pw50db = mean(sg_hat_pw50db(mk_csf),'all');
Mean.mk_wm.sg_hat_pw30db = mean(sg_hat_pw30db(mk_wm),'all');
Mean.mk_gm.sg_hat_pw30db = mean(sg_hat_pw30db(mk_gm),'all');
Mean.mk_csf.sg_hat_pw30db = mean(sg_hat_pw30db(mk_csf),'all');

Std.mk_wm.sg_h = std(sg_h(mk_wm),0,'all');
Std.mk_gm.sg_h = std(sg_h(mk_gm),0,'all');
Std.mk_csf.sg_h = std(sg_h(mk_csf),0,'all');
Std.mk_wm.sg_cr = std(sg_cr(mk_wm),0,'all');
Std.mk_gm.sg_cr = std(sg_cr(mk_gm),0,'all');
Std.mk_csf.sg_cr = std(sg_cr(mk_csf),0,'all');
Std.mk_wm.sg2 = std(sg2(mk_wm),0,'all');
Std.mk_gm.sg2 = std(sg2(mk_gm),0,'all');
Std.mk_csf.sg2 = std(sg2(mk_csf),0,'all');
Std.mk_wm.sg_hat_p = std(sg_hat_p(mk_wm),0,'all');
Std.mk_gm.sg_hat_p = std(sg_hat_p(mk_gm),0,'all');
Std.mk_csf.sg_hat_p = std(sg_hat_p(mk_csf),0,'all');
Std.mk_wm.sg_hat_pw = std(sg_hat_pw(mk_wm),0,'all');
Std.mk_gm.sg_hat_pw = std(sg_hat_pw(mk_gm),0,'all');
Std.mk_csf.sg_hat_pw = std(sg_hat_pw(mk_csf),0,'all');
Std.mk_wm.sg_hat_pw70db = std(sg_hat_pw70db(mk_wm),0,'all');
Std.mk_gm.sg_hat_pw70db = std(sg_hat_pw70db(mk_gm),0,'all');
Std.mk_csf.sg_hat_pw70db = std(sg_hat_pw70db(mk_csf),0,'all');
Std.mk_wm.sg_hat_pw50db = std(sg_hat_pw50db(mk_wm),0,'all');
Std.mk_gm.sg_hat_pw50db = std(sg_hat_pw50db(mk_gm),0,'all');
Std.mk_csf.sg_hat_pw50db = std(sg_hat_pw50db(mk_csf),0,'all');
Std.mk_wm.sg_hat_pw30db = std(sg_hat_pw30db(mk_wm),0,'all');
Std.mk_gm.sg_hat_pw30db = std(sg_hat_pw30db(mk_gm),0,'all');
Std.mk_csf.sg_hat_pw30db = std(sg_hat_pw30db(mk_csf),0,'all');


%% 
clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_onlyphase_\M1\coe\PINN_onlyphase34\TotalLoss.mat');
pt = 1:1:100;
pt0 = 1:length(loss)/100:length(loss);

loss = medfilt1(loss,50);
loss_pde = medfilt1(loss_pde,50);
loss_sg_bc = medfilt1(loss_sg_bc,50);
loss_w_sg = medfilt1(loss_w_sg,50);

figure('Name','LOSS');
subplot(2,2,1);set(gcf,'Color','w');
plot(pt,loss(pt0),'-','Color','#77AC30',LineWidth=1.5);
title('loss');
subplot(2,2,2);set(gcf,'Color','w');
plot(pt,loss_pde(pt0),'-','Color','#77AC30',LineWidth=1.5);
title('loss\_pde');
subplot(2,2,3);set(gcf,'Color','w');
plot(pt,loss_w_sg(pt0),'-','Color','#77AC30',LineWidth=1.5);
ylim(subplot(2,2,3),[0.2 0.7]);
title('loss\_w\_sg');
subplot(2,2,4);set(gcf,'Color','w');
plot(pt,loss_sg_bc(pt0),'-','Color','#77AC30',LineWidth=1.5);
ylim(subplot(2,2,4),[0.2 0.7]);
title('loss\_sg\_bc');


%% 
% ===================================================================================
% 确定损失函数权重
% ===================================================================================

clear all;close all;

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\ADEPT数据\Healthy\M1.mat');
slicestart = 75;
sliceend = 84;
sg = Conductivity_GT(:,:,slicestart:sliceend);
mk = sg ~= 0;
se = strel('cube',13);
se1 = strel('cube',3);
i1 = reshape(mk,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
i = imerode(i1,se,"same");
mk = reshape(i,size(sg));
sg = sg .* mk;

mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;

filepath = 'D:\ALL\科研\文章\my\20250526\result\M1\coe\';
file = dir(filepath);
file = file(3:end);
filenum = length(file);

% 提取文件名
filenames = {file.name};

% 提取IM后面的数字部分
% 假设所有文件名都符合IM+数字的格式
numbers = zeros(1, length(filenames));
for i = 1:length(filenames)
    % 使用正则表达式提取数字部分
    tokens = regexp(filenames{i}, 'PINN\_onlyphase(\d+)', 'tokens');
    if ~isempty(tokens)
        numbers(i) = str2double(tokens{1}{1});
    else
        % 如果文件名不符合预期格式，可以设为NaN或处理
        numbers(i) = NaN;
    end
end

% 按数字排序索引
[~, sortIdx] = sort(numbers);

% 使用排序索引重新排序文件名或结构体
sorted_files = file(sortIdx);
RD_ = zeros(length(file));
for i = 1:filenum
    
    filename = sorted_files(i).name;
    subfile = dir(fullfile(filepath,filename));
    subpath = fullfile(filepath,filename);
    load(fullfile(subpath,'sg.mat'));
    
    sg_hat = rot90(sg_hat,1);
    sg_hat = medfilt3(sg_hat,[3 3 3]);
    % im = reshape(sg_hat,[length(mk(:,1,1)),length(mk(1,:,1))*length(mk(1,1,:))]);
    % sg_hat = reshape(imerode(im,se1,"same"),size(mk));

    Mean.mk_wm.(filename) = mean(sg_hat(mk_wm),'all');
    Mean.mk_gm.(filename) = mean(sg_hat(mk_gm),'all');
    Mean.mk_csf.(filename) = mean(sg_hat(mk_csf),'all');
    Std.mk_wm.(filename) = std(sg_hat(mk_wm),0,'all');
    Std.mk_gm.(filename) = std(sg_hat(mk_gm),0,'all');
    Std.mk_csf.(filename) = std(sg_hat(mk_csf),0,'all');
    RD.mk_wm.(filename) = abs((Mean.mk_wm.(filename) - 0.35)*100./0.35);
    RD.mk_gm.(filename) = abs((Mean.mk_gm.(filename) - 0.69)*100./0.69);
    RD.mk_csf.(filename) = abs((Mean.mk_csf.(filename) - 2.26)*100./2.26);
    % RD.mk_wm.(filename) = mean(abs((sg_hat(mk_wm) - 0.35)*100./0.35),'all');
    % RD.mk_gm.(filename) = mean(abs((sg_hat(mk_gm) - 0.69)*100./0.69),'all');
    % RD.mk_csf.(filename) = mean(abs((sg_hat(mk_csf) - 2.26)*100./2.26),'all');
    RD.sum.(filename) = RD.mk_wm.(filename) + RD.mk_gm.(filename) + RD.mk_csf.(filename);
    RD_(i) = RD.sum.(filename);
end

pt = (0:1:length(file)-1)*0.1;
figure;set(gcf,'color','w');
plot(pt,RD_,'k--o');
legend('relative error');
legend('boxoff');
ylim([20 200]);
xlim([0 3.1]);


%% 
clear all;close all;
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\pinn_onlyphase.mat');
l_phase = gxx_phase + gyy_phase + gzz_phase;
figure;set(gcf,'color','w');
subplot(2,3,1);
imshow(rot90(TRx_phase(:,:,80),3),[0 4]);colorbar;colormap(turbo0white);
title('TRx\_phase');
subplot(2,3,2);
imshow(rot90(gx_phase(:,:,80),3),[-80 50]);colorbar;colormap(turbo0white);
title('gx\_phase');
subplot(2,3,3);
imshow(rot90(gy_phase(:,:,80),3),[-50 50]);colorbar;colormap(turbo0white);
title('gy\_phase');
subplot(2,3,4);
imshow(rot90(gz_phase(:,:,80),3),[0 50]);colorbar;colormap(turbo0white);
title('gz\_phase');
subplot(2,3,5);
imshow(rot90(l_phase(:,:,80),3),[-5e3 5e3]);colorbar;colormap(turbo0white);
title('l\_phase');

%% 
clear all;close all;
load('D:\ALL\科研\文章\my\20250526\phase.mat');
load('D:\ALL\科研\文章\my\20250526\ScanData\DICOM\PA1\ST0\phantom.mat');
cim_phantom = cim;
load('D:\ALL\科研\文章\my\20250526\ScanData\DICOM\PA1\ST0\brain_kspace_full.mat');
cim_brain = cim;

mk_phantom = abs(cim_phantom(:,:,:,1)) > 0.2;
mk_phantom_ = mk_phantom;
se = strel('cube',3);
i1 = reshape(mk_phantom_,[length(mk_phantom(:,1,1)),length(mk_phantom(1,:,1))*length(mk_phantom(1,1,:))]);
i = imopen(i1,se);
mk_phantom = reshape(i,size(mk_phantom));

mk_brain = abs(cim_brain(:,:,:,1)) > 0.2;
mk_brain_ = mk_brain;
se = strel('cube',13);
i1 = reshape(mk_brain_,[length(mk_brain(:,1,1)),length(mk_brain(1,:,1))*length(mk_brain(1,1,:))]);
% i = imopen(i1,se);
% se = strel('cube',13);
i = imclose(i1,se);
mk_brain = reshape(i,size(mk_brain));

im23_phantom = im23_phantom / 180 * pi;
im23_brain = im23_brain / 180 * pi;
w = 2*pi*128e6;
u0 = 4e-7*pi;
% l_phase_phantom = convn(im23_phantom,fspecial3('laplacian'),'same')/(1e-3)^2;
gxx_phase_phantom = convn(im23_phantom,[1 -2 1],'same') / (1e-3)^2;
gyy_phase_phantom = convn(im23_phantom,[1 -2 1]','same') / (1e-3)^2;
gzz_phase_phantom = convn(im23_phantom,permute([1 -2 1],[1 3 2]),'same') / (2e-3)^2;
l_phase_phantom = gxx_phase_phantom + gyy_phase_phantom + gzz_phase_phantom;
sg_h = l_phase_phantom./(2*w*u0);

figure;imshow(sg_h(:,:,30).*mk_phantom(:,:,30),[0 3]);colorbar;colormap turbo;

gxx_phase_brain = convn(im23_brain,[1 -2 1],'same') / (1e-3)^2;
gyy_phase_brain = convn(im23_brain,[1 -2 1]','same') / (1e-3)^2;
gzz_phase_brain = convn(im23_brain,permute([1 -2 1],[1 3 2]),'same') / (2e-3)^2;
l_phase_brain = gxx_phase_brain + gyy_phase_brain + gzz_phase_brain;
sg_h = l_phase_brain./(2*w*u0);

figure;imshow(sg_h(:,:,30).*mk_brain(:,:,60),[0 3]);colorbar;colormap turbo;

im23_phantom = im23_phantom .* mk_phantom;
im23_phantom(im23_phantom == 0) = nan;
im23_brain = im23_brain .* mk_brain;
im23_brain(im23_brain == 0) = nan;

% temp = zeros(size(im23_phantom));
% h5create('im23_phantom.h5','/trx_phase',size(im23_phantom));
% h5write('im23_phantom.h5','/trx_phase',im23_phantom);
% h5create('im23_phantom.h5','/tx_sens',size(im23_phantom));
% h5write('im23_phantom.h5','/tx_sens',temp);
% h5create('im23_phantom.h5','/mask_wm',size(im23_phantom));
% h5write('im23_phantom.h5','/mask_wm',temp);
% h5create('im23_phantom.h5','/sigma',size(im23_phantom));
% h5write('im23_phantom.h5','/sigma',temp);

temp = zeros(size(im23_brain));
h5create('im23_brain.h5','/trx_phase',size(im23_brain));
h5write('im23_brain.h5','/trx_phase',im23_brain);
h5create('im23_brain.h5','/tx_sens',size(im23_brain));
h5write('im23_brain.h5','/tx_sens',temp);
h5create('im23_brain.h5','/mask_wm',size(im23_brain));
h5write('im23_brain.h5','/mask_wm',temp);
h5create('im23_brain.h5','/sigma',size(im23_brain));
h5write('im23_brain.h5','/sigma',temp);

%% fig3

clear all;close all;
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_mam_3D_6_\PINN_mam_3D_6_31\TotalLoss.mat');
pt0 = 1:length(loss)/200:length(loss);
pt = 0:199;

figure('Name','LOSS');
subplot(2,2,1);set(gcf,'Color','w');
plot(pt,loss(pt0),'-','Color','#77AC30',LineWidth=1.5);
title('loss');
subplot(2,2,2);set(gcf,'Color','w');
plot(pt,loss_pde(pt0),'-','Color','#77AC30',LineWidth=1.5);
title('loss\_pde');
subplot(2,2,3);set(gcf,'Color','w');
plot(pt,loss_sg_bc(pt0),'-','Color','#77AC30',LineWidth=1.5);
title('loss\_sg\_w');
subplot(2,2,4);set(gcf,'Color','w');
plot(pt,loss_ep_bc(pt0),'-','Color','#77AC30',LineWidth=1.5);
title('loss\_ep\_w');


%% label2

clear all;close all;

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\matlab_code_and_data\pinn_data_3d.mat');
load('D:\ALL\SCM\MATLAB\colormap\turbo0white.mat');
slicestart = 75;
sliceend = 84;
ystart = 40;
yend = 209;
xstart = 65;
xend = 194;
slice1 = 3;
snr = 100;
sg = conductivity(xstart:xend,ystart:yend,slicestart:sliceend);
sg = rot90(sg,3);
ep = permittivity(xstart:xend,ystart:yend,slicestart:sliceend);
ep = rot90(ep,3);
mk_wm = abs(sg - 0.35) < 1e-4;
mk_gm = abs(sg - 0.69) < 1e-4;
mk_csf = abs(sg - 2.26) < 1e-4;

% figure(1);
% subplot(1,3,1);set(gcf,'Color','w');
% imshow(mk_wm(:,:,3),[0 1],'InitialMagnification','fit');colorbar;colormap(turbo0white);
% subplot(1,3,2);set(gcf,'Color','w');
% imshow(mk_gm(:,:,3),[0 1],'InitialMagnification','fit');colorbar;colormap(turbo0white);
% subplot(1,3,3);set(gcf,'Color','w');
% imshow(mk_csf(:,:,3),[0 1],'InitialMagnification','fit');colorbar;colormap(turbo0white);

water_content = zeros(size(sg));
water_content(abs(sg - 0.35) < 1e-4) = 0.7038;   % 0.6957
water_content(abs(sg - 0.69) < 1e-4) = 0.8593;   % 0.8341
water_content(abs(sg - 2.26) < 1e-4) = 0.9931;   % 0.9880
water_content = water_content + water_content / snr .* randn(size(water_content));
c1 = 0.286;
c2 = 1.526*1e-5;
c3 = 11.852;
p1 = -287;
p2 = 591;
p3 = -220;
sg1 = c1 + c2 * exp(c3 * water_content);
sg1(abs(sg1 - c1) < 1e-4) = 0;
% sg1 = sg1 * 1.15;
ep1 = p1 * water_content.^2 + p2 * water_content + p3;
ep1(abs(ep1 - p3) < 1e-4) = 1;
% ep1(ep1 ~= 1) = ep1(ep1 ~= 1) * 1.15;

cv = 0.25;

sg1_wm = sg1 .* mk_wm;
ep1_wm = ep1 .* mk_wm;
sg1_gm = sg1 .* mk_gm;
ep1_gm = ep1 .* mk_gm;
sg1_csf = sg1 .* mk_csf;
ep1_csf = ep1 .* mk_csf;
Mean.sg1_wm = mean(sg1_wm(sg1_wm~=0));
Mean.ep1_wm = mean(ep1_wm(ep1_wm~=0));
Mean.sg1_gm = mean(sg1_gm(sg1_gm~=0));
Mean.ep1_gm = mean(ep1_gm(ep1_gm~=0));
Mean.sg1_csf = mean(sg1_csf(sg1_csf~=0));
Mean.ep1_csf = mean(ep1_csf(ep1_csf~=0));
% Std.sg1_wm = std(sg1_wm(sg1_wm~=0));
% Std.ep1_wm = std(ep1_wm(ep1_wm~=0));
% Std.sg1_gm = std(sg1_gm(sg1_gm~=0));
% Std.ep1_gm = std(ep1_gm(ep1_gm~=0));
% Std.sg1_csf = std(sg1_csf(sg1_csf~=0));
% Std.ep1_csf = std(ep1_csf(ep1_csf~=0));

sg2 = zeros(size(sg1));
ep2 = ones(size(sg1));
sg2(sg1_wm~=0) = random('Normal',Mean.sg1_wm,Mean.sg1_wm*cv,size(sg1_wm(sg1_wm~=0)));
sg2(sg1_gm~=0) = random('Normal',Mean.sg1_gm,Mean.sg1_gm*cv,size(sg1_gm(sg1_gm~=0)));
sg2(sg1_csf~=0) = random('Normal',Mean.sg1_csf,Mean.sg1_csf*cv,size(sg1_csf(sg1_csf~=0)));
ep2(ep1_wm~=0) = random('Normal',Mean.ep1_wm,Mean.ep1_wm*cv,size(ep1_wm(ep1_wm~=0)));
ep2(ep1_gm~=0) = random('Normal',Mean.ep1_gm,Mean.ep1_gm*cv,size(ep1_gm(ep1_gm~=0)));
ep2(ep1_csf~=0) = random('Normal',Mean.ep1_csf,Mean.ep1_csf*cv,size(ep1_csf(ep1_csf~=0)));
sg2_wm = sg2 .* mk_wm;
ep2_wm = ep2 .* mk_wm;
sg2_gm = sg2 .* mk_gm;
ep2_gm = ep2 .* mk_gm;
sg2_csf = sg2 .* mk_csf;
ep2_csf = ep2 .* mk_csf;
Mean.sg2_wm = mean(sg2_wm(sg2_wm~=0));
Mean.ep2_wm = mean(ep2_wm(ep2_wm~=0));
Mean.sg2_gm = mean(sg2_gm(sg2_gm~=0));
Mean.ep2_gm = mean(ep2_gm(ep2_gm~=0));
Mean.sg2_csf = mean(sg2_csf(sg2_csf~=0));
Mean.ep2_csf = mean(ep2_csf(ep2_csf~=0));
Std.sg2_wm = std(sg2_wm(sg2_wm~=0));
Std.ep2_wm = std(ep2_wm(ep2_wm~=0));
Std.sg2_gm = std(sg2_gm(sg2_gm~=0));
Std.ep2_gm = std(ep2_gm(ep2_gm~=0));
Std.sg2_csf = std(sg2_csf(sg2_csf~=0));
Std.ep2_csf = std(ep2_csf(ep2_csf~=0));


load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_mam_3D_6_\PINN_mam_3D_6_41\epsc.mat');
sg_hat_inf = rot90(sg_hat(xstart:xend,ystart:yend,:),3);
ep_hat_inf = rot90(ep_hat(xstart:xend,ystart:yend,:),3);
sg_hat_inf_wm = sg_hat_inf .* mk_wm;
ep_hat_inf_wm = ep_hat_inf .* mk_wm;
sg_hat_inf_gm = sg_hat_inf .* mk_gm;
ep_hat_inf_gm = ep_hat_inf .* mk_gm;
sg_hat_inf_csf = sg_hat_inf .* mk_csf;
ep_hat_inf_csf = ep_hat_inf .* mk_csf;
Mean.sg_hat_inf_wm = mean(sg_hat_inf_wm(sg_hat_inf_wm~=0));
Mean.ep_hat_inf_wm = mean(ep_hat_inf_wm(ep_hat_inf_wm~=0));
Mean.sg_hat_inf_gm = mean(sg_hat_inf_gm(sg_hat_inf_gm~=0));
Mean.ep_hat_inf_gm = mean(ep_hat_inf_gm(ep_hat_inf_gm~=0));
Mean.sg_hat_inf_csf = mean(sg_hat_inf_csf(sg_hat_inf_csf~=0));
Mean.ep_hat_inf_csf = mean(ep_hat_inf_csf(ep_hat_inf_csf~=0));
Std.sg_hat_inf_wm = std(sg_hat_inf_wm(sg_hat_inf_wm~=0));
Std.ep_hat_inf_wm = std(ep_hat_inf_wm(ep_hat_inf_wm~=0));
Std.sg_hat_inf_gm = std(sg_hat_inf_gm(sg_hat_inf_gm~=0));
Std.ep_hat_inf_gm = std(ep_hat_inf_gm(ep_hat_inf_gm~=0));
Std.sg_hat_inf_csf = std(sg_hat_inf_csf(sg_hat_inf_csf~=0));
Std.ep_hat_inf_csf = std(ep_hat_inf_csf(ep_hat_inf_csf~=0));

load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_mam_3D_6_\PINN_mam_3D_6_42\epsc.mat');
sg_hat_100 = rot90(sg_hat(xstart:xend,ystart:yend,:),3);
ep_hat_100 = rot90(ep_hat(xstart:xend,ystart:yend,:),3);
sg_hat_100_wm = sg_hat_100 .* mk_wm;
ep_hat_100_wm = ep_hat_100 .* mk_wm;
sg_hat_100_gm = sg_hat_100 .* mk_gm;
ep_hat_100_gm = ep_hat_100 .* mk_gm;
sg_hat_100_csf = sg_hat_100 .* mk_csf;
ep_hat_100_csf = ep_hat_100 .* mk_csf;
Mean.sg_hat_100_wm = mean(sg_hat_100_wm(sg_hat_100_wm~=0));
Mean.ep_hat_100_wm = mean(ep_hat_100_wm(ep_hat_100_wm~=0));
Mean.sg_hat_100_gm = mean(sg_hat_100_gm(sg_hat_100_gm~=0));
Mean.ep_hat_100_gm = mean(ep_hat_100_gm(ep_hat_100_gm~=0));
Mean.sg_hat_100_csf = mean(sg_hat_100_csf(sg_hat_100_csf~=0));
Mean.ep_hat_100_csf = mean(ep_hat_100_csf(ep_hat_100_csf~=0));
Std.sg_hat_100_wm = std(sg_hat_100_wm(sg_hat_100_wm~=0));
Std.ep_hat_100_wm = std(ep_hat_100_wm(ep_hat_100_wm~=0));
Std.sg_hat_100_gm = std(sg_hat_100_gm(sg_hat_100_gm~=0));
Std.ep_hat_100_gm = std(ep_hat_100_gm(ep_hat_100_gm~=0));
Std.sg_hat_100_csf = std(sg_hat_100_csf(sg_hat_100_csf~=0));
Std.ep_hat_100_csf = std(ep_hat_100_csf(ep_hat_100_csf~=0));
load('D:\Pycharm\Project\PINN_modified_and_multiconstraints\PINN_modified_and_multiconstraints\result\image\PINN_mam_3D_6_\PINN_mam_3D_6_43\epsc.mat');
sg_hat_200 = rot90(sg_hat(xstart:xend,ystart:yend,:),3);
ep_hat_200 = rot90(ep_hat(xstart:xend,ystart:yend,:),3);
sg_hat_200_wm = sg_hat_200 .* mk_wm;
ep_hat_200_wm = ep_hat_200 .* mk_wm;
sg_hat_200_gm = sg_hat_200 .* mk_gm;
ep_hat_200_gm = ep_hat_200 .* mk_gm;
sg_hat_200_csf = sg_hat_200 .* mk_csf;
ep_hat_200_csf = ep_hat_200 .* mk_csf;
Mean.sg_hat_200_wm = mean(sg_hat_200_wm(sg_hat_200_wm~=0));
Mean.ep_hat_200_wm = mean(ep_hat_200_wm(ep_hat_200_wm~=0));
Mean.sg_hat_200_gm = mean(sg_hat_200_gm(sg_hat_200_gm~=0));
Mean.ep_hat_200_gm = mean(ep_hat_200_gm(ep_hat_200_gm~=0));
Mean.sg_hat_200_csf = mean(sg_hat_200_csf(sg_hat_200_csf~=0));
Mean.ep_hat_200_csf = mean(ep_hat_200_csf(ep_hat_200_csf~=0));
Std.sg_hat_200_wm = std(sg_hat_200_wm(sg_hat_200_wm~=0));
Std.ep_hat_200_wm = std(ep_hat_200_wm(ep_hat_200_wm~=0));
Std.sg_hat_200_gm = std(sg_hat_200_gm(sg_hat_200_gm~=0));
Std.ep_hat_200_gm = std(ep_hat_200_gm(ep_hat_200_gm~=0));
Std.sg_hat_200_csf = std(sg_hat_200_csf(sg_hat_200_csf~=0));
Std.ep_hat_200_csf = std(ep_hat_200_csf(ep_hat_200_csf~=0));


