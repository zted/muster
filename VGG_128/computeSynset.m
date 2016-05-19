function [Avg, Maxpool, Dispersion] = computeSynset(synsetId, tempDir, net)
%get a final vector representation for a given synset
%synsetId indicates the synsetId to compute (it must be a string)
%ImgsDir specifies the generic dir of the data set (e.g., imagenet)
%net is the CNN-128 model (pretrained)

Repr = []; Avg = []; Maxpool = [];

%find subfolders in the big one (not needed if we look for specific synsets)
%d = dir('C:/Guillem(work)/KU_Leuven/DATASETS/ImageNet_toy');
%isub = [d(:).isdir]; % returns logical vector
%nameFolds = {d(isub).name}';
%nameFolds(ismember(nameFolds,{'.','..'})) = []; %remove . and ..

%once we are in a given synset
%d3 = 'C:\Guillem(work)\KU_Leuven\DATASETS\ImageNet_toy'
%d4 = char(nameFolds(i)) %just in case we go through all folders
files = dir(fullfile(tempDir,char(synsetId))); %pick all the images
%files.name(ismember(files.name,{'.','..'})) = []; %remove . and ..
files = files(3:end); %delete the first two names of the folder . and ..
%files = dir(fullfile(ImgsDir,char(synsetId),'*.jpeg')); %pick just images
%with a certain extension e.g., JPEG
num_im = size(files,1); %get number of elements (images) in synset
max_im = num_im;

%We limit the number of elements to 1000 and minimum 50
if num_im >= 1000
    max_im = 1000;
elseif num_im < 50
    max_im = 0;
end

fn = fullfile(tempDir,char(synsetId))
if exist(fn, 'dir') == 7 %check whether the directory exists
    
    %main loop where we compute each image representation
    for j= 1:max_im
        %s = strcat(ImgsDir, '\', char(synsetId), '\',files(j).name); %get image j in the synset i
        s = strcat(tempDir, '/', char(synsetId), '/',files(j).name); %get image j in the synset i
        
        try
       
            im = imread(s);
            im_ = single(im) ; % note: 0-255 range
            im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
            im_ = im_ - net.meta.normalization.averageImage ;
            
            if size(im_,3)==3 %We take care of gray images (ignore them)
                %TODO: tryCatch
                % run the CNN
                res = vl_simplenn(net, im_) ;
                repr_im = res(20).x ; %for the vgg-128 network, to get the output of the last layer
                %repr_im = repmat(3,1,128); %just a fake vector
                
                Repr = [Repr; repr_im]; %add row representation
            end
        catch
            warning('Problem computing CNN. Assigning empty value');
        end
        
    end
    
    Avg = mean(Repr,1);
    Maxpool = max(Repr,[],1);
    Dispersion = calculateDispersion(Repr);
    %Entropy = 3
    
else
    warning('no directory')
end


end

%define local functions
function Disp = calculateDispersion(vecs)
%the input is a matrix vecs, where each representation is a row
numVecs = size(vecs,1);
accum = 0.0;
for i = 1:(numVecs-1),
    for j = (i+1):numVecs,
        vi = vecs(i,:); vj = vecs(j,:);
        dp = dot(vi,vj);
        denom = norm(vi) * norm(vj);
        accum = accum + (1-dp/denom);
    end
end
Disp = accum/(2.0*numVecs*(numVecs-1));
end



