function computeRepr()
%To run everything: 1.Open matlab in directory before the folder VGG_128
%2. codeDir = strcat(pwd, '/VGG_128'); addpath(codeDir);
%3.Type computeRepr in the matlab command

%IMPORTANT:
%1. Specify ImgsDir (where all folders are) and codeDir
%2. Specify directory vgg128Dir of pretrained model
%3. specify the words to get ('syns_and_words.txt'), or otherwise process all
%synsets (folders) from a given image dataset (set ALL = 1)
%4. specify a tempDir where the images from a synset are extracted and then
%eliminated

ALL = 0; %(0 = process just a subset, 1 = process all folders)

n_save = 50; % specifies every how many synsets we save INTERMEDIATE output

%specifies the generic dir of the data set (e.g., imagenet)
%codeDir = 'C:/Guillem(work)/KU_Leuven/Code/VGG_128'
%codeDir = strcat(pwd, '\VGG_128') %for windows
%ImgsDir = strcat(pwd, '\ImageNet_toy') %for windows
codeDir = strcat(pwd, '/VGG_128'); %for linux/Unix
%ImgsDir = strcat(pwd, '/ImageNet_toy'); %for linux/Unix
ImgsDir = '/media/guillem/d3d33e2b-6ce2-4851-a673-63546384fa55/extracted_imagenet';
tempDir = '/media/guillem';

addpath(codeDir);
addpath(tempDir);

% setup MatConvNet
run /home/guillem/MatConvNet/matlab/vl_setupnn
% load the pre-trained CNN (use the name you gave to this file after downloading)
vgg128Dir = '/media/guillem/vgg128.mat'
net = load(vgg128Dir) ;

if ALL == 0
    %We build a DICTIONARY from the words_to_get to synset_Ids
    syns_and_words = fopen('syns_and_words.txt','rt');
    C = textscan(syns_and_words,'%s %s','Delimiter',',','CollectOutput',1); %read csv
    fclose(syns_and_words);
    syns_and_words = C{1} %this is our list of tuples (synset_id, word)
    synsets = syns_and_words(:,1);
    wordlist = syns_and_words(:,2);
elseif ALL == 1
    %find subfolders in the big one (not when we look for specific synsets)
    d = dir(ImgsDir);
    isub = [d(:).isdir]; % returns logical vector
    nameFolds = {d(isub).name}';
    nameFolds(ismember(nameFolds,{'.','..'})) = []; %remove . and ..
    synsets = nameFolds;
    wordlist = nameFolds;
end

%write csv with all the words
fid = fopen('wordlist.csv', 'w') ;
fprintf(fid, '%s\n', wordlist{1:end}) ;
fclose(fid) ;

numsyns = size(synsets,1);
Avg = []; Maxpool = []; Dispersion = [];
words = {}; binary_words =  repmat(0,1,numsyns);
successes = 0;

%MAIN loop over all the synsets in words_to_get
for i = 1:numsyns
    
    synAvg = []; synMaxpool = []; synDispersion = []; %initialize them empty, to avoid problems
    
    try
        %create a temporary folder in tempDir with the synset
        tempFolder = strcat(tempDir, '/', char(synsets(i)));
        %mkdir(tempFolder); %not necessary
        %untar files into the new folder
        tarfilename = strcat(ImgsDir, '/', char(synsets(i)), '.tar');
        untar(tarfilename,tempFolder);
        
        %Compute the synset calculations
        [synAvg, synMaxpool, synDispersion] = computeSynset(synsets(i), tempDir, net);
        
        rmdir(tempFolder,'s') %remove the temporary folder we just created
        
    catch
        warning('tar file of the synset not found');
    end
    
    if isempty(synAvg) == 0 %just add representation when we have computed it
        binary_words(i) = 1; %if representation was computed, assign 1 to that word
        successes = successes +1;
        %Append the next vector
        words{successes} =  wordlist(i);
        Avg = [Avg ; synAvg];
        Maxpool = [Maxpool; synMaxpool];
        Dispersion = [Dispersion; synDispersion];
    end
    
    % write INTERMEDIATE OUTPUT every n_save files
    if floor(i/n_save)==i/n_save %just show it as progressbar and store INTERMEDIATE RESULTS
        fprintf(1,'******** File: %d our of %d ******\n',i,numsyns )
        save('saved','Avg','Maxpool','Dispersion', 'words','binary_words');
        try
            csvwrite('Avg.csv',Avg);
            csvwrite('Maxpool.csv',Maxpool);
            csvwrite('Dispersion.csv',Dispersion);
            csvwrite('binary_words.csv',binary_words);
            %write csv of words
            words2 = [words{:}];
            fid = fopen('words.csv', 'w') ;
            fprintf(fid, '%s\n', words2{1:end}) ;
            fclose(fid) ;
        catch
            warning('Format problem with writing the csv file');
        end
    end
    
end %end for

%once we have computed all the matrices, we generate the csv's files
save('saved_final','Avg','Maxpool','Dispersion', 'words', 'binary_words');

csvwrite('Avg.csv',Avg);
csvwrite('Maxpool.csv',Maxpool);
csvwrite('Dispersion.csv',Dispersion);
csvwrite('binary_words.csv',binary_words);
%write csv of words
words2 = [words{:}];
fid = fopen('words.csv', 'w') ;
fprintf(fid, '%s\n', words2{1:end}) ;
fclose(fid) ;

%try
%    writeCSV(words, Dispersion, 'dispersion.csv')
%    writeCSV(words, Avg, 'Avg.csv')
%    writeCSV(words, Maxpool, 'Maxpool.csv')
%catch
%    warning('Format problem with writing the csv file');
%end



 
 