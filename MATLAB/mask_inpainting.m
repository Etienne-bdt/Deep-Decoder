function [mask] = mask_inpainting(img)
    img = double(img);
    mask(:,:) = img(:,:,1)==1 & img(:,:,2)==1 & img(:,:,3) == 1;
    mask = double(mask);
    mask(:,:,2) = mask(:,:,1);
    mask(:,:,3) = mask(:,:,1);
end