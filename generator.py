from torch import nn

flatten_dim = 256*256*3

def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        im_dim: the dimension of the images 256*256 acts as noise vector
    '''
    def __init__(self, im_dim=flatten_dim, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            #in flattened image of dimension 3*(256**2) out 128
            get_generator_block(im_dim, hidden_dim), 
            #in 128 out 256
            get_generator_block(hidden_dim, hidden_dim * 2), 
            #in 256 out 512
            get_generator_block(hidden_dim * 2, hidden_dim * 4), 
            #in 512 out 1024
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            #in 1024, out flattened image
            nn.Linear(hidden_dim*8, im_dim), 
            #scale pixel intensities to between 0 and 1
            nn.Sigmoid() 
        )
    def forward(self, image):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor (photos), 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        
        return self.gen(image)
    