%define local functions
function writeCSV(words, matrix, filename)
%it receives a matrix (numerical) as input. Vector representations are rows
%the names input is the first column (in CELL format), i.e., the WORDS
%filename is a string that contains the name of the 'file.csv' (or .txt)
fid = fopen(filename, 'w') ;
for i = 1:size(matrix,1)
    fprintf(fid, '%s,', words{i}) ;
    fprintf(fid, repmat('%f,',1,size(matrix,2)), matrix(i,:)) ;
    fprintf(fid, '\n') ;
end
fclose(fid) ;
end



