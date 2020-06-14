    %%
%% A BIT ABOUT THIS PROGRAM:
%%% WE ARE TO COMPARE THE MEDIAN FILTERING AND GAUSSIAN FILTERING ON AN
%%% IMAGE. PROGRAM HAS BEEN CONFIGURED TO WORK BOTH WITH COLORED(RGB)
%%% IMAGE OR BLACK AND WHITE(GRAYSCALE) IMAGE.
%%% THE USER NEEDS TO FIRST INPUT THE TYPE OF NOISE TO BE ADDED TO THE
%%% IMAGE. THE KERNEL SIZE IS TAKEN AS 5 HERE. THE OUTPUT WILL DISPLAY 3
%%% WINDOWS. FIRST ONE DISPLAYS HOW GAUSSIAN FILTERING TAKES PLACE. SECOND
%%% ONE DISPLAYS HOW MEDIAN FILTERING TAKES PLACE. AND THE THIRD COMPARES
%%% THE GAUSSIAN AND MEDIAN FILTERED IMAGES.
%% BASIC CLEAR INSTRUCTIONS
clc;
close all;

%% TAKING THE IMAGE TO BE FILTERED
img = imread('Gaussian2.png');
%img = imread('Gaussian1.jpeg'); % Use this image for Gaussian Noise
% img = imread('Gaussian_noise.jpg');

%% INPUT THE TYPE OF NOISE
t = input('Enter 1- Gaussian Noise, 2- Salt and Pepper Noise\n    '); % for 1 
% we take the Gaussian noise and for 2 we take the salt and pepper noise
tic();

if t == 1
    %z = input('Enter the size of Kernel  ');
    z = 5;
    p1 = input('Enter Gaussian noise 1st parameter   ');
    p2 = input('Enter Gaussian noise 2nd parameter   ');

%Checking RGB or Gray scale and adding noise
if size(img,3) == 3
    I1 = rgb2gray(img);
    I = imnoise(I1, 'Gaussian', p1, p2);
    I2 = I;
else
    I = imnoise(img, 'Gaussian', p1, p2);
    I2 = I;
end

%Copy of the original image with noise
img3 = imnoise(img, 'Gaussian', p1, p2);

%% Gaussian Filtering
%%% IN GAUSSIAN FILTERING WE NEED TO MAKE A KERNEL AND THEN APPLY THAT
%%% KERNEL TO THE FILTER. IN STANDARD LIBRARY FILES THIS KERNEL IS ALREADY
%%% PRESENT. BUT SINCE WE ARE DOING THIS FROM SCRATCH SO WE MAKE OUR OWN
%%% KERNEL

% Making a Gaussian Kernel
sigma = 1;                     % standard deviation of the kernel
kernel = zeros(z,z);           %for a zXz kernel
W = 0;                         % sum of all elements of a kernel for normalization
for i = 1:5
    for j = 1:5
        sq_dist = (i-3)^2 + (j-3)^2;
        kernel(i,j) = exp(-1*(sq_dist)/(2*sigma*sigma));
        W = W + kernel(i,j);
    end
end
kernel = kernel/W;

% Applying the filter to the image
[m,n] = size(I);
output = zeros(m,n);
Im = padarray(I,[2 2]);

for i = 1:m
    for j = 1:n
        temp = Im(i:i+4 , j:j+4);
        temp = double(temp);
        conv = temp.*kernel;
        output(i,j) = sum(conv(:));
    end
    
end

output = uint8(output);

%% Median Filtering
%%% IN MEDIAN FILTERING WE DO NOT NEED TO CREATE A KERNEL AS IN THE
%%% PREVIOUS CASE. WE CAN DIRECTLY START APPLYING THE FILTER TO THE IMAGE.

[m1,n1] = size(I2);
output1 = zeros(m1,n1);
output1 = uint8(output1);

for a = 1:m1
    for b = 1:n1             % intensity of pixel in the noisy image is given as (i,j)
        xmin = max(1,a-1);   % minimum values of x y coordinate any pixel can take
        xmax = min(m1,a+1);
        ymin = max(1,b-1);
        ymax = min(n1,b+1);
        
        % Neighbourhood matrix will then be
        temp1 = I2(xmin:xmax, ymin:ymax);
        % The new intensity of pixel at (i,j) will be the medeian of this matrix
        output1(a,b) = median(temp1(:));
            
    end
end

elseif t == 2
        p2 = input('Enter Salt and Pepper noise parameter   ');
        %z = input('Enter the size of Kernel  ');

% Checking RGB or Gray scale and adding noise
if size(img,3) == 3
    I1 = rgb2gray(img);
    I = imnoise(I1, 'salt & pepper', p2);
    I2 = I;
else
    I = imnoise(img, 'salt & pepper', p2);
    I2 = I;
end

%Copy of the original image with noise
img3 = imnoise(img, 'salt & pepper', p2);

%% Gaussian Filtering

%Making a Gaussian Kernel
sigma = 1;                   % standard deviation of the kernel
kernel = zeros(z,z);         % for a zXz kernel
W = 0;                       % sum of all elements of a kernel for normalization
for i = 1:5
    for j = 1:5
        sq_dist = (i-3)^2 + (j-3)^2;
        kernel(i,j) = exp(-1*(sq_dist)/(2*sigma*sigma));
        W = W + kernel(i,j);
    end
end
kernel = kernel/W;

% Applying the filter to the image
[m,n] = size(I);
output = zeros(m,n);
Im = padarray(I,[2 2]);

for i = 1:m
    for j = 1:n
        temp = Im(i:i+4 , j:j+4);
        temp = double(temp);
        conv = temp.*kernel;
        output(i,j) = sum(conv(:));
    end
    
end

output = uint8(output);

%% Median Filtering
[m1,n1] = size(I2);
output1 = zeros(m1,n1);
output1 = uint8(output1);

for a = 1:m1
    for b = 1:n1             % intensity of pixel in the noisy image is given as (i,j)
        xmin = max(1,a-1);   % minimum values of x y coordinate any pixel can take
        xmax = min(m1,a+1);
        ymin = max(1,b-1);
        ymax = min(n1,b+1);
        
        % Neighbourhood matrix will then be
        temp1 = I2(xmin:xmax, ymin:ymax);
        % The new intensity of pixel at (i,j)will be deian of this matrix
        output1(a,b) = median(temp1(:));
            
    end
end
       
   
else
    disp('Incorrect Entry');
    
end

%% OUTPUT
%Showing original, noisy and filtered images side by side
figure(1);
set(gcf,'Position',get(0,'Screensize'));
subplot(131), imshow(img), title('Original Image');
subplot(132), imshow(img3), title('Original Image with Noise');
subplot(133), imshow(output), title('Output of Gaussian Filter');

figure(2);
set(gcf, 'Position', get(0,'Screensize'));
subplot(131), imshow(img), title('Original Image');
subplot(132), imshow(img3), title('Original Image with Noise');
subplot(133), imshow(output1), title('Output of Median Filter');

figure(3);
set(gcf, 'Position', get(0,'Screensize'));
subplot(121), imshow(output), title('Output of Gaussian Filter');
subplot(122), imshow(output1), title('Output of Median Filter');

%% TIME TAKEN FOR EXECUTION OF PROGRAM
elapsed = toc();
fprintf('Calculation took %.2f sec.\n', elapsed );%/ 60.0);
disp('THANK YOU!!');

%% THANK YOU!!
%%


        