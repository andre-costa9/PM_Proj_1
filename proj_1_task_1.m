filename= 'data.csv';

opts.DataRange = '6:';
data = readtable(filename,opts);

%aceder à info https://www.mathworks.com/matlabcentral/answers/405089-reading-the-data-from-a-csv-file-with-headers

wall_x= [];
wall_y= [];
d('job done %d \n',height(data));
%for i=1:1:
