# KeystrokeDynamic

Keystroke dynamics is the process of analyzing the way a user types at a terminal by monitoring the keyboard inputs thousands of times per second in an attempt to identify users based on habitual typing rhythm patterns. It has already been shown that keystroke rhythm is a good sign of identity. Moreover, unlike other biometric systems which may be expensive to implement, keystroke dynamics is almost free — the only hardware required is the keyboard. The application of keystroke rhythm to computer access security is relatively new. There has been some sporadic work done in this arena.

# 1.1 DATA  EXTRACTION  & SELECTION

We have recorded the keystroke features for each user in a file. The different Keystroke features which typing a series of characters, we have taken under consideration are, the time the subject needs to find the right key (flight time) and the time we holds down a key (dwell time) and the release time which is specific to that subject. So in that file we have in total 31 columns for each row and each row tuple corresponds to one complete password or data entry of a user in a single typing series.  


# 1.2 WEIGHT INITIALIZATION

In our Neural Network we have taken two hidden layers and each hidden layer has about 30 nodes. Weight initialization of egdes of network is done stepwise.Initially the Weight of input to hidden edges , hidden to hidden edges and hidden to output edges  was taken randomly for very first stage of learning .Then after completion of one complete epoch we saved the changed weights to a file so that when the next tuple of data was feeded then the weights was assigned according to the saved weights. Doing this we can improve the accuracy of our algorithm and this process take less time while running second time and after. Weight initialized is between the range of 0-1(both inclusive).

# 1.3 FEED FORWARD

Input to Hidden layer

We have converted each keystroke feature into binary form and then feeded into the network. Now for each node in the hidden layer, the hyperbolic tan function is applied to the sum of the products of the weight and the inputs for each node.If the value is above some threshold (typically 0) the neuron fires and takes the activated value (typically 1); otherwise it takes the deactivated value (typically -1). Neurons with this kind of activation function are also called artificial neurons or linear threshold units. In feedforward process we can use sigmod function but we instead use  hyperbolic tan function. 

hiddenVal1(i)=trainInputs(i) * weightsIH(i)

hiddenVal1(x)=F(x)

where F(X)= 1-e-2x/1+e-2x

Hidden to Hidden layer

Similar for each node in the second hidden layer the hyperbolic tan function is applied to the sum of the product of the weights of the edges between the nodes of the two layers and the calculated result of tan function in the previous hidden layer.

hiddenVal2(i)=hiddenVal1(i) * weightsHH(i)

hiddenVal2(x)=F(x)

where F(X)= 1-e-2x/1+e-2x

Hidden to output layer
Now the sum of product of result of tan function of each node in the second hidden layer, with the weights connecting the edges with the nodes of the output layer is calculated. Which is the actual ouput for the given input in the Neural Network.
Actual Output = hiddenVal2(i) * weightsHO(i)

# 1.4 COMPUTE ERROR

Error is calculated from the difference of Actual trained network output  and Expected network output.

Error= (Actual Network Output – Expected Network Ouput)

# 1.5 BACK PROPAGATION 

The method calculates the gradient of the error function with respect to the neural network's weights. In this process we back propagate from output to input and change the weigts value according to corresponding RMS error calculated from expected output and actual output. First we change the weight value of edges of  output node to hidden node and then we change weight value of the edges connecting the one hidden layer to the another hidden layer and lastly for the edges connecting the hidden layer and the input layer. while changing weight there we used two thing learning rate which may vary from its application but its value is taken very small between 0 and 1. 

Weight change from output to hidden :

weightChange = LR_HO * errThisPat * hiddenval2(i)

Weight change from hidden to Hidden:

weightChange =hiddenVal1(i)*weightsHH[i] * errThisPat * LR_HH*(1 –hiddenVal2(i)^2)

Weight change from hidden to Input:

weightChange =trainInputs(i)*weightsHO[i] * errThisPat * LR_IH*(1 -hiddenVal(i)^2)

# 1.6 WEIGHT UPDATE

The weights of input-hidden, hidden-hidden and  hidden-output edges according to weigt changes formula is updated for each edge in all the layers. This weight updation and backpropagation process is continued until our neural network is trained upto the desired precision level.
