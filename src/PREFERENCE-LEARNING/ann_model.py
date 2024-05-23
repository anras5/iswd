import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
from threshold_layer import ThresholdLayer


class LinearGreaterThanZero(nn.Linear):
    """Linear layer with weights constrained to be greater than zero.
    """
    def __init__(self,in_features:int, bias:bool=False, min_w:float=0.0000001):
        """Constructor for LinearGreaterThanZero

        Args:
            in_features (int): Number of input features/ criteria
            bias (bool, optional): Whether to include bias. Defaults to False.
            min_w (float, optional): Minimum value for the weights. Defaults to 0.0000001.
        """
        super().__init__(in_features,1, bias)
        self.is_bias = bias
        self.min_w = min_w
        if bias:
            nn.init.uniform_(self.bias, self.min_w ,1.)
        else:
            self.bias = None

    def reset_parameters(self):
        """Reset the weights of the layer"""
        nn.init.uniform_(self.weight, 0.1,1.)


    def w(self)->torch.Tensor:
        """Get the weights of the layer and apply the constraint that the weights 
            should be greater than zero.

        Returns:
            torch.Tensor: Weights of the layer
        """
        with torch.no_grad():
            self.weight.data[self.weight.data<0]=self.min_w
        return self.weight

    def forward(self, input:torch.Tensor)->torch.Tensor:
        """Forward pass of the layer

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return F.linear(input, self.w(), self.bias)


class LinearInteraction(nn.Linear):
    """Linear layer for criteria interaction. It contains constraints that the inpact of each criteria should be positive.
    """
    def __init__(self,in_features:int, criterion_layer:LinearGreaterThanZero):
        """Constructor for LinearInteraction

        Args:
            in_features (int): Number of input features/ criteria.
            criterion_layer (LinearGreaterThanZero): Layer with the criteria weights.
        """
        super().__init__(((in_features-1)*in_features)//2,1, False)
        self.in_features = in_features
        self.criterion_layer = criterion_layer
        
    def reset_parameters(self):
        """Reset the weights of the layer.
        """
        nn.init.normal_(self.weight, 0.0,0.1)     
    
    def w(self)->torch.Tensor:
        """Get the weights of the layer and apply the constraint that the weights.

        Returns:
            torch.Tensor: Weights of the layer.
        """
        with torch.no_grad():
            w_i=0
            w = self.criterion_layer.w()
            for i in range(self.in_features):
                for j in range(i+1,self.in_features):
                    self.weight.data[:,w_i] =  torch.max(self.weight.data[:,w_i], -w[:,i])
                    self.weight.data[:,w_i] =  torch.max(self.weight.data[:,w_i], -w[:,j])
                    w_i+=1
        return self.weight
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        """Forward pass of the layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return F.linear(input, self.w(), None)


class ChoquetIntegralConstrained(nn.Module):
    """Choquet Integral model with constraints on the weights.
    """
    def __init__(self, num_criteria:int):
        """Constructor for ChoquetIntegralConstrained

        Args:
            num_criteria (int): Number of criteria.
        """
        super().__init__()
        self.num_criteria = num_criteria
        self.criteria_layer = LinearGreaterThanZero(num_criteria)
        self.interaction_layer = LinearInteraction(num_criteria,self.criteria_layer)
        self.threshold_layer = ThresholdLayer()
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        """Forward pass of the model.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """        """"""    
        if len(input.shape)==3:
            input = input[:,0,:]
        
        # Calculate the criteria part of integral
        x_wi = self.criteria_layer(input[:,:self.num_criteria])     
        # Calculate the interaction part of integral 
        x_wij = self.interaction_layer(input[:,self.num_criteria:])
        # Normalize scores
        weight_sum = self.criteria_layer.w().sum()+self.interaction_layer.w().sum()
        score =  (x_wi+x_wij)/(weight_sum)
        return self.threshold_layer(score)


def mobious_transform(row:list|np.ndarray)->list:
    """Mobious transform of the input row.
    First n values are the criteria values and the rest are the minimum values of the pairs of criteria.

    Args:
        row (list | np.ndarray): Input row.

    Returns:
        list: Mobious transform of the input row.
    """
    return  list(row) + [min(row[i],row[j]) for i in range(len(row)) for j in range(i+1,len(row))]