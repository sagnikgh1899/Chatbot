%WORKING 

clc
pL = 0.0899;
pR = 0.0854;
F = 43;
b = 1.3;

camL = ipcam('http://192.168.1.101:8080/video');
camR = ipcam('http://192.168.1.100:8080/video');
 N = 1;
  for i = 1:N
 ARless = snapshot(camR);
     ALless = snapshot(camL);
     AL1less= imrotate(ALless,-180,'bilinear','crop');
  end
  
  AR = ARless + 60; %+ 121.5;
  AL1 = AL1less + 60;% + 105.5;
  
  AL1rmat = AL1(:,:,1);
AL1gmat = AL1(:,:,2);
AL1bmat = AL1(:,:,3);

levelr = 0.5;
levelg = 0.4;
levelb = 0.3;

AL11 = imbinarize(AL1rmat,levelr);
AL12 = imbinarize(AL1gmat,levelg);
AL13 = imbinarize(AL1bmat,levelb);
AL1Isum = (AL11&AL12&AL13);

AL1Icomp = imcomplement(AL1Isum);
AL1Ifilled = imfill(AL1Icomp,'holes');

seL2 = strel('disk',12);
 seL1 = strel('diamond',12);
 seL3 = strel('rectangle',[35 25]);
 
 AL1Iopenned1 = imopen(AL1Ifilled,seL1);
AL1Iopenned2 = imopen(AL1Iopenned1,seL3);
AL1Iopenned = imopen(AL1Iopenned2,seL2);

ARrmat = AR(:,:,1);
ARgmat = AR(:,:,2);
ARbmat = AR(:,:,3);

AR11 = imbinarize(ARrmat,levelr);
AR12 = imbinarize(ARgmat,levelg);
AR13 = imbinarize(ARbmat,levelb);
ARIsum = (AR11&AR12&AR13);

ARIcomp = imcomplement(ARIsum);
ARIfilled = imfill(ARIcomp,'holes');

RseL2 = strel('disk',12);
 RseL1 = strel('diamond',12);
 RseL3 = strel('rectangle',[35 25]);
 
 ARIopenned1 = imopen(ARIfilled,RseL1);
ARIopenned2 = imopen(AL1Iopenned1,RseL3);
ARIopenned = imopen(ARIopenned2,RseL2);

AL1Iregion = regionprops(AL1Iopenned, 'centroid');
AL1centroid = AL1Iregion;
[labeled,AL1numObjects] = bwlabel(AL1Iopenned,4);
AL1stats = regionprops(labeled,'Eccentricity', 'Area', 'BoundingBox');
AL1areas = [AL1stats.Area];
AL1eccentricities = [AL1stats.Eccentricity];

AL1idxOfSkittles = find(AL1eccentricities);
AL1statsDefects = AL1stats(AL1idxOfSkittles);


ARIregion = regionprops(ARIopenned, 'centroid');
ARcentroid = ARIregion;
[labeled,ARnumObjects] = bwlabel(ARIopenned,4);
ARstats = regionprops(labeled,'Eccentricity', 'Area', 'BoundingBox');
ARareas = [ARstats.Area];
AReccentricities = [ARstats.Eccentricity];

ARidxOfSkittles = find(AReccentricities);
ARstatsDefects = ARstats(ARidxOfSkittles);

figure, imshow(AL1less);
hold on;

if AL1numObjects==ARnumObjects
    
  for AL1idx = 1 : length(AL1idxOfSkittles)
    AL1h = rectangle('Position', AL1statsDefects(AL1idx).BoundingBox);
    set (AL1h,'EdgeColor' , [1 1 0]);
    hold on;
  end  
  
  % tic
x_centroid = zeros(1, 1);
y_centroid = zeros(1, 1);
xL = zeros(1, 1);
xR = zeros(1, 1);
yL = zeros(1, 1);
yR = zeros(1, 1);
x = zeros(1,1);
L = zeros(1,1);
L1 = zeros(1,1);

%   for i=1:length(AL1Iregion)
%     x_centroid(i) = AL1Iregion(i).Centroid(1);
%     y_centroid(i) = AL1Iregion(i).Centroid(2);
%   %end
%   % toc
%   
%   xL(i) = x_centroid(i) * pL;
%   yL(i) = y_centroid(i) * pL;
%   xR(i) = x_centroid(i) * pR;
%   yR(i) = y_centroid(i) * pR;
%   
%   x(i) = xL(i) - xR(i);
%   L1(i) = (F*b);
%   L(i) = L1(i)/x(i);
%     
%   end
  
  if AL1idx == 1
      for i=1:length(AL1Iregion)
    x_centroid(i) = AL1Iregion(i).Centroid(1);
    y_centroid(i) = AL1Iregion(i).Centroid(2);
  %end
  % toc
  
  xL(i) = x_centroid(i) * pL;
  yL(i) = y_centroid(i) * pL;
  xR(i) = x_centroid(i) * pR;
  yR(i) = y_centroid(i) * pR;
  
  x(i) = xL(i) - xR(i);
  L1(i) = (F*b);
  L(i) = L1(i)/x(i);
      end
     txt = ['Distance: ' num2str(L) ' cm'];
     text(220,130,txt,'FontSize',14,'EdgeColor','Red');
       % title(['There is ', num2str(AL1numObjects), ' Object in the image and ','Distance = ', num2str(L), 'cm']);
        %title(['Distance = ', num2str(L), 'cm']);
   
  end
  
  if AL1idx>1
       for i=1:length(AL1Iregion)
    x_centroid(i) = AL1Iregion(i).Centroid(1);
    y_centroid(i) = AL1Iregion(i).Centroid(2);
  %end
  % toc
  
  xL(i) = x_centroid(i) * pL;
  yL(i) = y_centroid(i) * pL;
  xR(i) = x_centroid(i) * pR;
  yR(i) = y_centroid(i) * pR;
  
  x(i) = xL(i) - xR(i);
  L1(i) = (F*b);
  L(i) = L1(i)/x(i);
  for j = 1:i
        txt(['Distance of object ', num2str(j), ' = ', num2str(L(i)), 'cm' ]);
        text(10,0.4,txt,'FontSize',14,'EdgeColor','Red');
  end
       end   
  end
  
end
hold off;




