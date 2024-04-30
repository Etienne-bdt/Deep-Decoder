function out = loss_inp(Y,T)
    arguments 
       Y dlarray,
       T dlarray,
    end

    load mask.mat mask;
    mask = ~mask;
    out = mse(Y.*mask,T.*mask);
end