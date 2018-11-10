function track_objects()

% this is a very simple tracking algo
% for more serious tracking, look-up the papers in the projects pdf
FRAME_DIR = '../data/frames/';
DET_DIR = '../data/detections/';
start_frame = 62;
end_frame = 71;
%tracks = zeros(0,13);   %formart [dets_cur dets_next occurences]

%field names
previous_frame = 'prev_frame';  previous_frame_val = zeros(4,0);  %had to transpose so ismember would work
occurences = 'occur';  occurences_val = {0};
dets = 'dets';  dets_val = zeros(0,6);
frames_field = 'frames'; frames_val = zeros(0,0);  % the frames a detection occurs in 
tracks = struct(previous_frame,previous_frame_val,occurences,occurences_val,dets,dets_val,frames_field,frames_val);


    for idx = start_frame:end_frame
        im_cur = imread(fullfile(FRAME_DIR, sprintf('%06d.jpg', idx)));
        data = load(fullfile(DET_DIR, sprintf('%06d_dets.mat', idx)));
        dets_cur = data.dets;

        im_next = imread(fullfile(FRAME_DIR, sprintf('%06d.jpg', idx+1)));
        data = load(fullfile(DET_DIR, sprintf('%06d_dets.mat', idx+1)));
        dets_next = data.dets;

        % sim has as many rows as dets_cur and as many columns as dets_next
        % sim(k,t) is similarity between detection k in frame i, and detection
        % t in frame j
        % sim(k,t)=0 means that k and t should probably not be the same track
        sim = compute_similarity(dets_cur, dets_next, im_cur, im_next);

        %greedy approach 
        finding = true;   %loop until no values in sim are > 0
        while finding
            [maxValue, idxs] = max(sim(:));  %find current max
            [i,j] = ind2sub(size(sim),idxs);  % get its i and j

            if maxValue == 0 
                finding = false; %stop loop all sims>0 are found
            else
                %get previous dets locations from tracks
                prev_dets = [tracks.prev_frame]'; % transpose so that ismember function could find row

                %see if current det is in tracks by comparing with prev_dets
                trans_dets_cur = dets_cur(i,1:4); % get current sims detection  box
                [Result,LocResult] = ismember(trans_dets_cur,prev_dets,'rows');

                if Result  %if current det is in tracks, then add to that track
                    curr_occurs = tracks(LocResult).occur + 1; %increase number of occurence for current track by 1
                    curr_dets = tracks(LocResult).dets;    
                    previous_frame_val = dets_next(j,1:4)';   %set prev_fram value to det_next 
                    dets_val = [curr_dets;dets_next(j,1:4)];  %append the next det to the tracks dets list
                    frames_val = [tracks(LocResult).frames idx+1];  %save the frames this track was seen in 
                    
                    tracks(LocResult) = struct(previous_frame,previous_frame_val,occurences,curr_occurs,dets,dets_val,frames_field,frames_val);
                    %tracks(LocResult,:) = [dets_cur(i,:) dets_next(j,:) curr_occurs+1];   %save the detections, occured twice (current & next)                             
                    
                else
                    %add current det and next det to a new track
                    track_size = size(tracks,2);      
                    curr_occurs = 2;  
                    previous_frame_val = dets_next(j,1:4)';
                    dets_val = [dets_cur(i,1:4);dets_next(j,1:4)];
                    frames_val = [idx idx+1];
                    
                    tracks(track_size+1) = struct(previous_frame,previous_frame_val,occurences,curr_occurs,dets,dets_val,frames_field,frames_val);
                    %tracks(track_size+1) = [dets_cur(i,:) dets_next(j,:) 2];   %save the detections, occured twice (current & next)         
                end
                sim(i,j)=0; %set the current sim to 0 so it wont be choosen again

            end
        end   

    end;

    
    %only take tracks with occurences > 2 
    idx = find([tracks.occur]>2);
    correct_tracks = tracks(idx);
    
    %visualize tracks with more than 5
    idx = find([tracks.occur]>5);
    visualize_tracks = tracks(idx);
    track_size = size(visualize_tracks,2);
    colors = {'r','g','b', 'y', 'm', 'c', 'w','k'}; %set colors for modified showboxes function
    
    %%loop through each frame and show detections 
    for idx = start_frame:end_frame
        im_cur = imread(fullfile(FRAME_DIR, sprintf('%06d.jpg', idx)));
        boxes = [];
        frame_colors = {};
        
        %get the boxes (loop through each track)
        for i = 1:track_size
            track_dets = visualize_tracks(i).dets; %get dets for all frams on current track
            track_frames = visualize_tracks(i).frames;
            location = find(track_frames == idx);
            
            if location  %if current track has current frame
                boxes = [boxes track_dets(location,:)];
                frame_colors{size(frame_colors,2)+1} = colors{i}; %set a color got current box
            end
        end
        
        figure,showboxes(im_cur, boxes, frame_colors) %run modified showbox function

    end
    
    
end


function sim = compute_similarity(dets_cur, dets_next, im_cur, im_next)

n = size(dets_cur, 1);
m = size(dets_next, 1);
sim = zeros(n, m);


area_cur = compute_area(dets_cur);
area_next = compute_area(dets_next);
c_cur = compute_center(dets_cur);
c_next = compute_center(dets_next);
im_cur = double(im_cur);
im_next = double(im_next);
weights = [1,1,2];

for i = 1: n
    % compare sizes of boxes
    a = area_cur(i) * ones(m, 1);
    sim(i, :) = sim(i, :) + weights(1) * (min(area_next, a) ./ max(area_next, a))';
    
    % penalize distance (would be good to look-up flow, but it's slow to
    % compute for images of this size)
    sim(i, :) = sim(i, :) + weights(2) * exp((-0.5*sum((repmat(c_cur(i, :), [size(c_next, 1), 1]) - c_next).^2, 2)) / 5^2)';
    
    % compute similarity of patches
    box = round(dets_cur(i, 1:4));
    box(1:2) = max([1,1],box(1:2));
    box(3:4) = [min(box(3),size(im_cur, 2)), min(box(4),size(im_cur, 1))];
    im_i = im_cur(box(2):box(4),box(1):box(3), :);
    im_i = im_i / norm(im_i(:));
    for j = 1 : m
       d = norm(c_cur(i, :) - c_next(j, :));
       if d>60  % distance between boxes too big
           sim(i,j) = 0;
           continue;
       end;
       box = round(dets_next(j, 1:4));
       box(1:2) = max([1,1],box(1:2));
       box(3:4) = [min(box(3),size(im_cur, 2)), min(box(4),size(im_cur, 1))]; 
       im_j = im_next(box(2):box(4),box(1):box(3), :);
       im_j = double(imresize(uint8(im_j), [size(im_i, 1), size(im_i, 2)]));
       im_j = im_j / norm(im_j(:));
       c = sum(im_i(:) .* im_j(:));
       sim(i,j) = sim(i,j) + weights(3) * c;
    end;
end;
end

function area = compute_area(dets)
   area = (dets(:, 3) - dets(:, 1) + 1).* (dets(:, 4) - dets(:, 2) + 1);
end

function c = compute_center(dets)

c = 0.5 * (dets(:, [1:2]) + dets(:, [3:4]));
end


