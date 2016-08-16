function W = randInitializeWeights(L_in, L_out)

% randInitializeWeights Randomly initialize the weights of a layer with L_in
% incoming connections and L_out outgoing connections so that we break the
% symmetry while training the neural network. 


% The first row of W corresponds to the parameters for the bias units

epsilon_init=sqrt(6)/sqrt(L_in+L_out);
W = rand(L_out, 1 + L_in)*2*epsilon_init-epsilon_init;



end
