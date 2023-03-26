function [state_estimate] = hack_init_estimates( markersn )

n_m = 8;
T = 0.1;
markersn = reshape(markersn', 3, 8, []);
markersn(markersn==1e10) = nan; 
centroids = squeeze(nanmean(markersn, 2)); 

% initialize COM position with centroid position
p = centroids(:,1);

% initialize COM velocity with centroid velocity (approximately)
v = (centroids(:,end) - centroids(:,1)) / ( (size(centroids,2)-1)*T ) ; 

state_estimate = [ p(1) p(2) p(3) v(1) v(2) v(3) 1 0 0 0 0 0 0 ]';