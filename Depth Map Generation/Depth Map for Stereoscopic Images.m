% %CODE BY - SAGNIK GHOSAL; 
% JADAVPUR UNIVERSITY
% ELECTRICAL ENGINEERING DEPARTMENT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause(2);
camR = ipcam('http://192.168.1.8:8080/video');
camL = ipcam('http://192.168.1.2:8080/video');

    N = 1;
 for i = 1:N
    AR = snapshot(camR);
    AL = snapshot(camL);
    AL1= imrotate(AL,-180,'bilinear','crop');
imwrite(AR,strcat('F:\prog matlab\R1',num2str(i),'.png'));
imwrite(AL1,strcat('F:\prog matlab\L1',num2str(i),'.png'));
   pause(5);
 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('newstereoparams1.mat');
I2 = imread('F:\prog matlab\L11.png');
I1 = imread('F:\prog matlab\R11.png');
[frameRightRect, frameLeftRect] = rectifyStereoImages(I1, I2,stereoParams1);

figure;
imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
title('Rectified Video Frames');
left=frameLeftRect;
right=frameRightRect;
left_I=mean(left,3);
right_I=mean(right,3);
I_disp=zeros(size(left_I),'single');
disp_range=32;
h_block_size=3;
blocksize=h_block_size*2+1;
row=size(left_I,1);
col=size(left_I,2);

for m =1:row
    row_min= max(1, m- h_block_size);
    row_max= min(row, m+ h_block_size);

    for n =1:col
          col_min= max(1,n-h_block_size);
          col_max= min(col,n+h_block_size);
          %% setting the pixel search limit

          pix_min= max(-disp_range, 1 - col_min);
          pix_max= min(disp_range, col - col_max);

          template = right_I(row_min:row_max ,col_min:col_max);

          block_count= pix_max - pix_min + 1;
          block_diff= zeros(block_count, 1);

          for i = pix_min : pix_max
              block= left_I(row_min:row_max ,(col_min +i ):(col_max+i));
              index= i-pix_min+1;
              block_diff(index,1)= sumsqr(template-block);

          end
          [B,I]= sort(block_diff);
          match_index= I(1,1);
          disparity= match_index+pix_min-1;
          I_disp(m, n) = disparity;
    end
end

imshow(I_disp);
colormap jet;
colorbar ;
caxis([0 16]);
title('Depth map from  block matching');
colormapeditor

points3D = reconstructScene(I_disp, stereoParams1);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);