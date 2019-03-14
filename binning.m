addpath('Universal_directed_information')
data = load('data/spikeTimes_medium.mat');
dbin = data.data;

index = find([dbin(:,:)]>=2) % make sure binning does not double up
numneu = 50

DImat = zeros(numneu,numneu);
MImat = zeros(numneu,numneu);
for n=1:numneu
    for m = 1:numneu
	disp([n,m])
        x = dbin(n, :);
        y = dbin(m, :);
        [MI, DI, rev_DI] = compute_DI_MI(x, y, 2, 2, 'E4', 0, 0, 0);
        DImat(n,m) = DI(length(DI));
        MImat(n,m) = MI(length(MI));
    end
end

save('data/spikeTimes_medium_DI.mat', 'DImat')
save('data/spikeTimes_medium_MI.mat', 'MImat')

exit;
