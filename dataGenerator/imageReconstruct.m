function c = imageReconstruct(b,K,alpha,QmapLR)

K1         =  blockfun(mean(b,3),[K K],@mean);
c          =  K^2/alpha*gammaincinv(1-K1,QmapLR,'upper');
c          =  max(min(c,1),0);