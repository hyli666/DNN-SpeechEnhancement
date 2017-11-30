function [x] = ISTFT(X)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
franum=size(X,2);
Xful=zeros(512,franum);
nw=512;
nm=512/2;
ys=zeros(nw,franum);
for ind=1:franum
    Xful(1:257,ind)=X(:,ind);
    Xful(258:end,ind)=conj(X(256:-1:2,ind));
    ys(:,ind)=real(ifft(Xful(:,ind)));
end
ys2=zeros(nw,1);
ix=0;
clear x;
for ind=1:franum
    ys2 = ys(:,ind) + [ys2(nm+1:end); zeros(nm,1)];
    x(ix+1:ix+nm)=ys2(1:nm);
    ix=ix+nm;
end
x=x(256:end)';
end

