data = load('data/spikeTimes_medium.mat');
dbin = data.data;
dbin = [];

index = find([dbin(:,:)]>=2) % make sure binning does not double up

size(dbin)
numneu = 50;

DImat = zeros(numneu,numneu);
MImat = zeros(numneu,numneu);
for n=1:numneu
    for m = 1:numneu
        x = dbin(n, :);
        y = dbin(m, :);
        [MI, DI, rev_DI] = compute_DI_MI(x, y, 2, 2, 'E4', 0, 0, 0);
        DImat(n,m) = DI(length(DI));
        MImat(n,m) = MI(length(MI));
    end
end

DImat