clear;
close all;
tic
% the image size
img_size=[512,512];
% grid_size=[16,16];
% grid_spacing=ceil([img_size(1)/grid_size(1)+2,img_size(2)/grid_size(2)+2]);

% gaussian function for gaussian speckle
mymodel=@(x,y,r)200*exp(-((x).^2+(y).^2)/r^2);
[X,Y]=meshgrid(single(-256:255),single(-256:255));
save_path='/home/lanshihai/program code/Lan/traning dataset/';
[xx_p,yy_p]=meshgrid(1:540,1:540);
num = 100;
for img_num=1:num
    % the radius of the speckle
    r=1.2;
    
    speckle_random=0.5;
    
    % count of the speckle
    num_speckle=0;
    
    % grid size, grid spacing for gaussian speckle image generation
    grid_size=3*randi([5,7]);
    grid_spacing=floor([img_size(1)/grid_size(1),img_size(2)/grid_size(2)]);
    
    % aver_speckle_num in a grid
    aver_speckle_num=randi([4,7]);

    for i=1:grid_spacing(1)
        for j=1:grid_spacing(2)
            
            grid_x=i*grid_size(1);
            grid_y=j*grid_size(2);
            
            % the count of the speckle in a single grid
            num_grid_speckle=ceil(abs(normrnd(aver_speckle_num,0.6)));
            for k=1:num_grid_speckle
                num_speckle=num_speckle+1;
                
                % the location of the gaussian speckle
                speckle_x(num_speckle)=grid_x+2*grid_size(1)*rand()-grid_size(1);
                speckle_y(num_speckle)=grid_y+2*grid_size(2)*rand()-grid_size(2);
            end
            
        end
    end
    speckle_x=single(speckle_x);
    speckle_y=single(speckle_y);
    re_gray_img=single(zeros(512,512));
    tar_gray_img=single(zeros(512,512));
    displacment_x=single(zeros(540,540));
    displacment_y=single(zeros(540,540));
    
    % the grid size of the random displacement generation
    dis_grid=12*randi([2,5]);
    grid_num=ceil(540/dis_grid)+1;
    
    % the standard devitaion of the random displacement 
    dis_aver_x=rand()*1.5+1;
    dis_aver_y=rand()*1.5+1;
    f=normrnd(0,dis_aver_x,[grid_num,grid_num]);
    g=normrnd(0,dis_aver_y,[grid_num,grid_num]);
    [x_p,y_p]=meshgrid(1:dis_grid:dis_grid*(grid_num-1)+1,1:dis_grid:dis_grid*(grid_num-1)+1);
    
    % the added displacement in the pixel location
    displacment_x=displacment_x+single(interp2(x_p,y_p,f,xx_p,yy_p,'spline'));
    displacment_y=displacment_y+single(interp2(x_p,y_p,g,xx_p,yy_p,'spline'));
    
    % translation
    translation_x=(rand()-0.5)*6;
    translation_y=(rand()-0.5)*6;
    
    % total displacement
    displacment_x=displacment_x+translation_x;
    displacment_y=displacment_y+translation_y;
    
%     figure;
%     imagesc(displacment_x(15:526,15:526));
%     colormap jet;
%     colorbar;
%     figure;
%     imagesc(displacment_y(15:526,15:526));
%     colormap jet;
%     colorbar;

    
    % speckle location after deformation
    speckle_dis_x=single(interp2(xx_p,yy_p,displacment_x,speckle_x,speckle_y,'spline'));
    speckle_dis_y=single(interp2(xx_p,yy_p,displacment_y,speckle_x,speckle_y,'spline'));
    
    % generate the reference and target image 
    for i=1:num_speckle
        
        re_gray_img=re_gray_img+mymodel(X-speckle_x(i)+270,Y-speckle_y(i)+270,r);
        tar_gray_img=tar_gray_img+mymodel(X-speckle_x(i)-speckle_dis_x(i)+270,Y-speckle_y(i)-speckle_dis_y(i)+270,r);
    end
    
%     figure;
%     imshow(re_gray_img,[]);
%     figure;
%     imshow(tar_gray_img,[]);
    
    % noise addition
    re_gray_img=imnoise(uint8(re_gray_img),'gaussian',0,(0.01)^2);
    tar_gray_img=imnoise(uint8(tar_gray_img),'gaussian',0,(0.01)^2);
    
    % save the speckle image and displacement field
    imwrite(uint8(re_gray_img),[save_path,'re',num2str(img_num,'%04d'),'.bmp'],'bmp');
    imwrite(uint8(tar_gray_img),[save_path,'tar',num2str(img_num,'%04d'),'.bmp'],'bmp');
    csvwrite([save_path,'u',num2str(img_num,'%04d'),'.csv'],displacment_x(15:526,15:526));
    csvwrite([save_path,'v',num2str(img_num,'%04d'),'.csv'],displacment_y(15:526,15:526));
end

