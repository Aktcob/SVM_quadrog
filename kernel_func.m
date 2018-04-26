function K = kernel_func( X, Y, type, P)
if type==1
    K=X'*Y;
else
    K=(X'*Y + 1).^P;
end    
end