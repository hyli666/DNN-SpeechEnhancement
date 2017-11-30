function [X] = STFT(x)
%输入信号x，输出STFT矩阵X
x=x(:);              % 输入信号的列向量化

nw = 512;             % 帧长
nm = 512/2;                          % 帧移
win = hamming(nw+1);
win(end) = []; win = win(:);        % 窗函数
win = win/(0.54*nw/nm);             % 抵消窗交叠重构后的增益

in=0;
xs = zeros(nw,1);                   % 帧数据（输入）
ind=1;
while(in+nm<=length(x))              % 判断需要更新帧的数据是否超出输入信号(x)长度，否则跳出循环
    xs = [xs(nm+1:end);x(in+1:in+nm)];% 取x的nm个点与xs的nw-nm个点组成新的xs，实现了数据帧

    XS = fft(xs.*win);              % 求xs加窗后的傅立叶变换
    X(:,ind)=XS(1:nw/2+1);
    in=in+nm;
    ind=ind+1;
end


end

