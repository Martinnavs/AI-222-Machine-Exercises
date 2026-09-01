My initial implementation plan:
1. create a DataClass first to parse the MNIST images into torch tensors first.
2. then implement the read function to parse the MNIST images into torch tensors. And then given the MNIST image dimension, educate me on how to identify if the logic needs padding or not.

3 (parse this). create the convolution logic by creating a function that takes a kernel size and filter size. Teach me on how this is implemented, particularly on how to choose a kernel size (mentioned in the lectures as having small pixel dimensions from 2-4) and filter size (this hasn't been discussed clearly in the lecture). What I know is that a convolution is basically a matmul using the following base loop per channel:


precondition: initialize the kernel output tensor given the dimensions calculated using the kernel size, stride size, and maxpool against the image size

starting from [0,0]:
- locate the mxn chunk and do a matmul with the kernel. store this as the ith pixel in the output tensor.
- move to s (as in stride) pixels forward (right movement, down movement if reaches the end)

what is currently not clear here is if we need to manually do the parallelization here or if einsum and/or torch already handles it per channel. In addition, since each filter serves as separate convolutions, will there be a loop above the per-channel loop to do this?

- In addition, since this is a 3-layer CNN, I assume that there will be 3 convolution operations utilizing the created function but with different kernel and potentially stride lengths. educate me on whether varying the kernel and stride are usually altered across layers.

4. lastly, flatten the last layer into something that can be acceptable into a dense network (which we will also implement here.)

For the dense network, it's as simple as taking the last layer as the input nodes and pass it through the next layers via matmul(w.T, x) + b (which will be written in einops). Lastly, since this is a classification problem (with 10 classes), we use a sigmoid function and then get the highest probability as the classified class.

What's not clear to me here is:
1. How do we pick the sizes of the next layers? are they the same until the pre-output node where the last node is x dimensions (10 in this case due to MNIST)?
2. Do we apply nonlinearity between the layers? if so can we use simpler funcs such as ReLU and the use Sigmoid in the last layer?