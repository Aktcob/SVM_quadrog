function K = new_kernel( X, Y, m1, m2, P)
    K=(m1*(X'*Y)+m2).^P;   
end