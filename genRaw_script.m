for kk=1:23
step=40;
start1=cell(5,1);
end1=cell(5,1);

parfor ii=1:5
    start1{ii}=(ii-1)*step+1+(kk-1)*200;
    end1{ii}=start1{ii}+step-1;
    genRawData_func(start1{ii},end1{ii},ii+(kk-1)*5);
end

clear start1 end1

end