function [w, perc_error, crit, time] = FB_sto(winit, X_mat, Y, lambda, delta, p, theta, d_test, Y_test, ItMax, Stop_norm, Stop_crit)

display_it = 2000 ;

L = length(Y) ;
L_test = length(Y_test) ;
w = winit;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO COMPLETE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions g and h
g=@(w) lambda * norm(w,1);
h=@(w) (1/L) * sum(huber(Y - X_mat.' * w, delta));

% approximated gradient of smooth function
% compute only gradient wrt functions associated to indices given by Ind 
grad_par =@(w, Ind) (-1/size(Ind,2)) * X_mat(:,Ind) * huber_grad(Y(Ind) - X_mat(:,Ind).' * w, delta);

% proximity operator of non-smooth function
prox =@(w, T) max(abs(w)-T, 0).*sign(w);
   
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for it = 1:ItMax
    % Select randomly a subset of indices corresponding to the functions
    % for which we compute the gradient
    Ind_it = sort(randperm(L, floor(L*p))) ;
    wold = w; 
    
    t_start = tic;
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TO COMPLETE
    % stochastic prox gradient descent iterations
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gamma = 1/((it + 1)^theta); % step-size
    w = prox(wold - gamma * grad_par(wold, Ind_it), lambda * gamma);
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    time(it) = toc(t_start) ;
    
    norm_w(it) = norm(w-wold)/norm(w) ;
    crit(it) = g(w) + h(w) ;
    diff = abs(Y_test - d_test(w))/2 ;
    perc_error(it) = sum(diff)/L_test*100 ;
    
    if mod(it,display_it)==0
        disp(['Iteration ', num2str(it)])
        disp(['Time = ', num2str(sum(time))])
        disp(['error (%) on test set = ', num2str(perc_error(it))])
        disp(['crit = ', num2str(crit(it))])
        disp(['relative norm iterates = ', num2str(norm_w(it))])
        disp('****************************************')
        
        figure(101)
        subplot 131, plot(perc_error), xlabel('it'), ylabel('error (%)'), axis([0 it+1 0 100])
        subplot 132, semilogy(crit(1:end-1)-crit(2:end)), xlabel('it'), ylabel('$f(x_k) - f(x_{k-1})$', 'Interpreter', 'latex')
        subplot 133, semilogy(norm_w), xlabel('it'), ylabel('$\| x_k - x_{k-1} \| / \|x_k\|$', 'Interpreter', 'latex')
        pause(0.1)
    end
    
    if it >10 ...
            && abs(crit(it)-crit(it-1))/abs(crit(it)) < Stop_crit ...
            && norm_w(it) < Stop_norm
        break
    end
end

disp('****************************************')
disp(['STOP Iteration ', num2str(it)])
disp(['Time = ', num2str(sum(time))])
disp(['error (%) on test set = ', num2str(perc_error(it))])
disp(['crit = ', num2str(crit(it))])
disp(['relative norm iterates = ', num2str(norm_w(it))])
disp('****************************************')



end