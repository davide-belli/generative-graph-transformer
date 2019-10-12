import torch
from torch import nn
import torch.nn.functional as F


class CNNEncoderSimple(nn.Module):
    r"""
    Simple CNN encoder with 2 convolutional layers , max-pooling and 2 linear layers.
    Uses Leaky-ReLU and batch_norm.
    
    Input: [B x 1 x 64 x 64] grayscale image
    Output: [B x 900] conditioning vector
    """
    def __init__(self):
        super(CNNEncoderSimple, self).__init__()
        # input_dim  # b, 1, 64, 64
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)  # b, 8, 64, 64
        self.pool1 = nn.MaxPool2d(2, stride=2)  # b, 8, 32, 32
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=0)  # b, 16, 30, 30
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 900, 4 * 900)  # b, 4 * 900
        self.fc1_bn = nn.BatchNorm1d(4 * 900)
        self.fc2 = nn.Linear(4 * 900, 900)  # b, 900
    
    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x))
        x = self.conv1_bn(self.pool1(x))
        x = self.conv2_bn(F.leaky_relu_(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1_bn(F.leaky_relu_(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNNDecoderSimple(nn.Module):
    r"""
    Simple CNN decoder with 2 linear layers and 2 trasposed convolutional layers and 2 linear layers.
    Uses Leaky-ReLU and batch_norm.

    Input: [B x 900] conditioning vector
    Output: [B x 1 x 64 x 64] grayscale image
    """
    def __init__(self):
        super(CNNDecoderSimple, self).__init__()
        # input_dim  # b, 900
        self.fc1 = nn.Linear(900, 4 * 900)  # b, 4 * 30 * 30
        self.fc1_bn = nn.BatchNorm1d(4 * 900)
        self.fc2 = nn.Linear(4 * 900, 32 * 16 * 16)  # b, 32 * 16 * 16
        self.fc2_bn = nn.BatchNorm1d(32 * 16 * 16)
        self.conv1 = nn.ConvTranspose2d(32, 8, 4, stride=2, padding=1)  # b, 8, 32, 32
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)  # b, 1, 64, 64
    
    def forward(self, x):
        x = self.fc1_bn(F.leaky_relu_(self.fc1(x)))
        x = self.fc2_bn(F.leaky_relu_(self.fc2(x)))
        x = x.reshape(x.shape[0], 32, 16, 16)
        x = self.conv1_bn(F.leaky_relu_(self.conv1(x)))
        x = nn.Tanh()(self.conv2(x))
        
        return x


class CNNEncoderSimpleForContextAttention(nn.Module):
    r"""
    Simple CNN encoder with 3 convolutional layers and max-pooling, used in the context attention encoder
    Uses Leaky-ReLU and batch_norm.

    Input: [B x 1 x 64 x 64] grayscale image
    Output: [B x 900] conditioning vector
    """
    def __init__(self):
        super(CNNEncoderSimpleForContextAttention, self).__init__()
        # input_dim  # b, 1, 64, 64
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)  # b, 8, 64, 64
        self.pool1 = nn.MaxPool2d(2, stride=2)  # b, 8, 32, 32
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=0)  # b, 16, 30, 30
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 1, stride=1, padding=0)  # b, 1, 30, 30
    
    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x))
        x = self.conv1_bn(self.pool1(x))
        x = self.conv2_bn(F.leaky_relu_(self.conv2(x)))
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        return x
    

class CNNEncoderAtt(nn.Module):
    r"""
    CNN encoder with context attention over the image features.
    First, uses the CNNEncoderSimpleForContextAttention to output flattened visual features.
    Then, an MLP and a softmax are used to generate a mask on the visual features, taking as input the representation
    of the graph generated so far (zero-padded) and the original visual features.
    The mask is applied to the visual features by element-wise product.

    Input: [B x 1 x 64 x 64] grayscale image
            [B x SEQ_LEN x SEQ_LEN * MAX_PREV_NODE] partial adjacency matrix at each timestep in the sequence
            [B x SEQ_LEN x SEQ_LEN * 2] node feature matrix at each timestep in the sequence
    Output: [B x 900] conditioning vector
    """
    def __init__(self, adj_size=4, coord_size=2, visual_size=900):
        super(CNNEncoderAtt, self).__init__()
        # self.cnn = CNNEncoderSimple()
        self.cnn = CNNEncoderSimpleForContextAttention()
        self.linear_att1 = nn.Linear(coord_size + adj_size + visual_size, visual_size * 2)
        self.linear_att2 = nn.Linear(visual_size * 2, visual_size)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, img, x_coord, x_adj, get_att_weights=False):
        img = self.cnn(img)
        img = img.unsqueeze(1).repeat(1, x_adj.shape[1], 1)
        input_sequence = torch.cat([x_coord, x_adj, img], dim=2)
        input_sequence = F.relu_(self.linear_att1(input_sequence))
        input_sequence = self.linear_att2(input_sequence)
        mask = self.softmax(input_sequence)
        masked_img = mask * img
        
        # optionally, return the attention weights of the context attention
        if get_att_weights:
            mask = mask[0].reshape(1, 30, 30)
            return masked_img, mask[0]
        
        return masked_img


if __name__ == '__main__':
    encoder = CNNEncoderSimple().to(device="cuda:0")
    decoder = CNNDecoderSimple().to(device="cuda:0")
    x = torch.rand(8, 1, 64, 64).to(device="cuda:0")
    z = encoder(x)
    x_hat = decoder(z)
    print(z.shape)
    print(x_hat.shape)
