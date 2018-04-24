%NUMBERS = [2, 6, 10, 50, 70, 100];
NUMBERS = [20, 25, 60, 50, 40, 30, 35, 40, 377, 443, 507, 433];
SAVE_FOLDER = '/is/ps2/yhuang2/Projects/ESMPLify/Data/DCT_Basis';

for num = NUMBERS
    res_path = fullfile(SAVE_FOLDER, num2str(num));
    if exist(res_path, 'file')
       continue;
    else
       D = dctmtx(num); 
       save(res_path, 'D');
    end    
end
