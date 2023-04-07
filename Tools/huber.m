function h=huber(x, delta)
h = zeros(size(x)) ;
h(x<-delta) = delta* (abs(x(x<-delta)) - delta/2) ;
h(x>delta) = delta* (abs(x(x>delta)) - delta/2) ;
h(abs(x)<=delta) = x((abs(x)<=delta)).^2/ 2 ;
end

