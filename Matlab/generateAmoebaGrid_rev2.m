function generateAmoebaGrid_rev2(L_plate,N_plate)
% % % the lengths approx. are:
% % % Violin: 45 cm
% % % Viola: 40cm
% % % Cello: 152cm
% % % Pontic Lyre: 61cm
% % % Cretan Lyre: 51cm 

% L_plate = 0.8 * 0.4/0.6200; % for Viola
% L_plate = 1 * 0.51/0.82; % for Cretan Lyre
% L_plate = 1 * 0.59/0.7143;
% L_plate = 1 * 0.8;

% N_plate = 19; % Fix the number of points on orig grid
% N_plate = 50; % Fix the number of points on orig grid
% N_plate = 168; % Fix the number of points on orig grid
% N_plate = 192;
% N_plate = 100;

h_plate = L_plate/N_plate;

% Select points for analysis (circular grid)
xVec = [0:N_plate-1];
yVec = [0:N_plate-1];

%     [X,Y] = meshgrid(xVec,yVec);
[X,Y] = ndgrid(xVec,yVec); % This is the correct way ! 


filesPath = cd;
filesPath = fullfile(filesPath, '.\Amoeba_Pics');

% Get coordinates from violin image
I = imread(fullfile(filesPath,'Amoeba_bitmap_rev2.jpg'));

% figure;
% imshow(I)

greenChannel = I(:, :, 2);
binaryImage = greenChannel < 50;

binaryImage = ~binaryImage;

locStart = find(binaryImage == 0,1,'first');
[~,col_start] = ind2sub(size(binaryImage),locStart);
locEnd = find(binaryImage == 0,1,'last');
[~,col_end] = ind2sub(size(binaryImage),locEnd);

binaryImage_flip = binaryImage.';

locStart = find(binaryImage_flip == 0,1,'first');
[~,row_start] = ind2sub(size(binaryImage_flip),locStart);
locEnd = find(binaryImage_flip == 0,1,'last');
[~,row_end] = ind2sub(size(binaryImage_flip),locEnd);

binaryImage = binaryImage(row_start:row_end,col_start:col_end);

% figure;
% imshow(binaryImage)

binaryBuild = ones(max(size(binaryImage))+80,max(size(binaryImage))+80);

placementX = floor(size(binaryBuild,1)/2)-floor(size(binaryImage,1)/2);
placementY = floor(size(binaryBuild,2)/2)-floor(size(binaryImage,2)/2);
binaryBuild(placementX:placementX + size(binaryImage,1) - 1,placementY:placementY + size(binaryImage,2) - 1) = binaryImage;

% figure;
% imshow(binaryBuild)

% % adding some extra rows/columns
% scaleAdjustX = 30;
% scaleAdjustY = 30;    
% binaryImage = cat(2,ones(size(binaryImage,1),scaleAdjustX),binaryImage,ones(size(binaryImage,1),scaleAdjustX));
% binaryImage = cat(1,ones(scaleAdjustY,size(binaryImage,2)),binaryImage,ones(scaleAdjustY,size(binaryImage,2)));

% figure;
% imshow(binaryImage)


binaryImage = ~binaryBuild;

N_scale = N_plate;

newSize = [N_scale,N_scale];
scaleFactors = [N_scale/size(binaryImage,1),N_scale/size(binaryImage,2)];

newImage = imresize(binaryImage,'Scale',scaleFactors);

% figure;
% imshow(newImage);

newImage = flip(newImage,2);    

ana = newImage(:);
locs = find(ana == 1);

% 
% figure(3);
% plot(X(:).*h_plate,Y(:).*h_plate,'x')
% grid on
% hold all
% % plot(xCirc,circle_y_1)
% % plot(xCirc,circle_y_2)
% axis equal
% plot(X(locs).*h_plate,Y(locs).*h_plate,'o')


locsDo = locs;
save(['locsAll_violin_size_',num2str(N_plate),'_x_',num2str(N_plate),'.mat'],'locsDo');

end
