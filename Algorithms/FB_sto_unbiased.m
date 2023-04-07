function [w, perc_error, crit, time] = FB_sto_unbiased(winit, X_mat, Y, lambda, delta, p, beta, d_test, Y_test, ItMax, Stop_norm, Stop_crit)

display_it = 2000 ;

L = length(Y) ;
L_test = length(Y_test) ;
w = winit;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO COMPLETE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions g and h
g=@(w) lambda * norm(w,1);
h=@(w) 1/L * sum(huber(Y - X_mat.' * w, delta)); % L = floor(L*p)
% approximated gradient of smooth function
% compute only gradient wrt functions associated to indices given by Ind 
grad_par =@(w, Ind) (-1/L) * X_mat(:,Ind) * huber_grad(Y(Ind) - X_mat(:,Ind).' * w, delta); %(-1 /L)
% proximity operator of non-smooth function
prox =@(w, T) max(abs(w)-T, 0).*sign(w);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create subsets of indices corresponding to the functions
% for which we compute the gradient
nb_ind = floor(L*p) ;
Ind_glob = zeros(nb_ind, 1/p) ;
Ind_perm = randperm(L) ;
for i = 1:1/p
    Ind_glob(:,i) = Ind_perm((i-1)*nb_ind+1:i*nb_ind) ;
    Ind_glob(:,i) = sort(Ind_glob(:,i)) ;
end
clear Ind_perm
grad_save = zeros(numel(w), 1/p) ;
grad = zeros(size(w));

for it = 1:ItMax
    ind_it = randperm(1/p, 1) ;
    Ind_it = Ind_glob(:,ind_it) ;
    wold = w; 
    
    t_start = tic;
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TO COMPLETE
    % stochastic prox gradient descent iterations
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gamma = 1 / beta ; 
    % w = ...

    % Compute the approximated gradient only for the selected indices
    grad_new = grad_par(w, Ind_it);    
    % Update the stored gradients for the current subset of indices
    grad_diff = grad_new - grad_save(:, ind_it);
    grad_save(:, ind_it) = grad_new;   
    % Update the unbiased gradient
    uk = uk + grad_diff;  
    % Update the model parameters
    w = prox(w - gamma * uk, lambda * gamma);
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