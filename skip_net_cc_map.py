import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
# import meshio
from in_wenjian_cc import write_in, read_grad, dis_xy, delete_doc
import os


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up_DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Skip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.skip(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect
        self.skip = Skip(out_channels, 4)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        if self.skip_connect:
            out1 = self.double_conv(x)
            out2 = self.skip(out1)
            return out1, out2
        else:
            out1 = self.double_conv(x)
            return out1


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv_1 = Up_DoubleConv(in_channels, out_channels, in_channels // 2)
        self.double_conv_2 = Up_DoubleConv(in_channels+4, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        out1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - out1.size()[2]
            diffX = x2.size()[3] - out1.size()[3]

            out1 = F.pad(out1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        if self.skip_connect:
            out2 = torch.cat([out1, x2], dim=1)
            out3 = self.double_conv_2(out2)
            return out3
        else:
            out4 = self.double_conv_1(out1)
            return out4


class AdaptSigmoid(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.log_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.coefficient = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight to 1.0.
        self.log_weight.data.fill_(1.0)
        self.coefficient.data.fill_(1.0)

    def forward(self, input):
        # Compute the parameterized sigmoid: 1 / (1 + exp(-weight * input)).
        weight = torch.sigmoid(self.log_weight)
        # print(f"Sigmoid parameter weight: {weight}")
        # coeff = torch.exp(self.coefficient)
        coeff = F.softplus(self.coefficient)
        # print(f"Sigmoid parameter weight: {weight}")
        # print(f"Sigmoid parameter coefficient: {coeff}")
        return coeff * torch.sigmoid(input) + weight


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.soft_plus = nn.Softplus()
        self.autoSigmoid = AdaptSigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.autoSigmoid(x)
        return x


def idw_interp(points1, values1, points2, p=2):
    """
    points1: Array of shape (4000, 2), containing normalized coordinates of circle 1.
    values1: Array of shape (4000,), containing nodal values of circle 1.
    points2: Array of shape (400, 2), containing normalized coordinates of circle 2.
    p: Distance exponent; defaults to 2.
    Returns: Array of shape (400,), containing nodal values of circle 2.
    """
    values2 = np.zeros(len(points2))
    dc_dm = np.zeros((len(points2), len(points1)))     # Jacobian matrix of the mapping.
    # print(f'Jacobian matrix: {dc_dm.shape}')
    for j in range(len(points2)):
        # Calculate the distances from point j to all source points i.
        d = np.linalg.norm(points1 - points2[j], axis=1)
        # print(d)
        # Avoid division by zero by replacing zero distances with a small epsilon.
        d = np.where(d == 0, 1e-10, d)
        # Calculate the weights.
        w = 1.0 / (d ** p)
        # print(w)
        # Calculate the weighted average.
        v_j = np.sum(w * values1) / np.sum(w)
        values2[j] = v_j
        dc_dm[j] = w / np.sum(w)
        # print(np.sum(w))
    # print(dc_dm.shape)
    return values2, dc_dm


def x_y_coord_sm(sm_mat, x_coord_mat, y_coord_mat ):
    y_coord_t = np.flip(y_coord_mat)
    # print(y_coord)
    total_coord_list = []
    total_sm_list = []
    # print(len(x_coord_mat))
    for i in range(len(x_coord_mat)):
        for j in range(len(y_coord_t)):
            coord_list = [x_coord_mat[j], y_coord_t[i]]
            total_coord_list.append(coord_list)
            total_sm_list.append(sm_mat[i, j])
    total_coord_array = np.array(total_coord_list)
    total_sm_array = np.array(total_sm_list)
    return total_coord_array, total_sm_array


class FEM(Function):

    @staticmethod
    def forward(self, input, np_coord, mat_x_coord, mat_y_coord):
        numpy_input_4 = input.detach().cpu().numpy()      # Predicted shear modulus distribution.
        numpy_input = np.squeeze(numpy_input_4)
        # ctx.save_for_backward(numpy_input)        # ctx.save_for_backward can only store Variables.

        nn_coord, nn_sm = x_y_coord_sm(numpy_input, mat_x_coord, mat_y_coord)
        # print(nn_coord.shape)
        # print(nn_sm.shape)

        fem_sm, jb_map = idw_interp(nn_coord, nn_sm, np_coord, p=3)
        # print(fem_sm.shape)
        self.jb_grad = jb_map  # Obtain the gradient of the coordinate mapping.
        # fem_sm_in = np.flip(fem_sm.reshape(48, 48), axis=0)
        # print(fem_sm.reshape(48, 48))
        # print(jb_map.shape)


        self.input_np = numpy_input_4
        write_in(fem_sm, 'cricle_case_150.in')
        os.system('./nlace.exe cricle_case_150.in')  # Compute the displacement field.
        result_x, result_y = dis_xy('inverse_cci0.vtk')       # Compute the displacement field.

        # print(f"Result: {result_x.sum()+result_y.sum()}")
        # print(input.new(result_x).shape)
        result_dis = torch.stack([input.new(result_x), input.new(result_y)])  # Combine x and y displacements in a tensor with the same type as input.
        return result_dis

    @staticmethod
    def backward(self, grad_output):  # Compute the gradient manually with NumPy.
        print(f"loss反向梯度大小是{grad_output.shape}")
        input_sm = self.input_np
        jb_grad = self.jb_grad
        print(input_sm.shape)

        grad_np = read_grad('inverse_cci0.vtk')
        delete_doc()
        grad_fem_to_nn = np.dot(grad_np, jb_grad)
        result = np.expand_dims(grad_fem_to_nn.reshape(input_sm.shape[2], input_sm.shape[3]), axis=(0, 1))
        print(f"fem反向梯度大小是{result.shape}")
        # g = grad_output.new(result) * grad_output
        print(f"grad_output.new形状{grad_output.new(result).shape}")
        # print(f"Backward gradient shape: {g.shape}")
        grad_coord = None
        grad_x = None
        grad_y = None
        return grad_output.new(result), grad_coord, grad_x, grad_y


class SkipNet(nn.Module):
    def __init__(self, n_channels, out_channels, np_coord, mat_x_coord, mat_y_coord):
        super().__init__()
        self.down1 = Down(n_channels, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32, skip_connect=True)
        self.down4 = Down(32, 64, skip_connect=True)
        self.down5 = Down(64, 128)
        self.up1 = Up(128, 64, skip_connect=True)
        self.up2 = Up(64, 32, skip_connect=True)
        self.up3 = Up(32, 16)
        self.up4 = Up(16, 8)
        self.up5 = Up(8, 8)
        self.last_conv = LastConv(8, out_channels)
        self.np_coord = np_coord
        self.mat_x_coord = mat_x_coord
        self.mat_y_coord = mat_y_coord

    def forward(self, x):
        x1 = self.down1(x)
        print(x1.size())
        x2 = self.down2(x1)
        x3, s_x1 = self.down3(x2)
        print(x3.size())
        x4, s_x2 = self.down4(x3)
        print(x4.size())
        x5 = self.down5(x4)
        print(x5.size())
        x6 = self.up1(x5, s_x2)
        print(x6.size())
        x7 = self.up2(x6, s_x1)
        x8 = self.up3(x7, None)
        x9 = self.up4(x8, None)
        x10 = self.up5(x9, None)
        x11 = self.last_conv(x10)
        print(x11.size())
        out_dis = FEM.apply(x11, self.np_coord, self.mat_x_coord, self.mat_y_coord)
        print(out_dis.size())
        return out_dis, x11


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Apply Kaiming initialization.
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            # Initialize the bias to a constant.
            nn.init.constant_(m.bias, 0)



if __name__ == "__main__":
    # mode = DoubleConv(1, 8)
    inp = torch.randn((1, 1, 128, 128))
    fem_coord = np.loadtxt('fem_coord.txt', delimiter=',')
    def mat_up_low(fem_coord_f, x_node, y_node):
        upper_limit = np.max(np.ceil(np.max(fem_coord_f, axis=0)))
        lower_limit = np.min(np.floor(np.min(fem_coord_f, axis=0)))
        print(f"upper_limit:{upper_limit}")
        print(f"lower_limit:{lower_limit}")
        x_coord_f = np.linspace(lower_limit, upper_limit, x_node)
        y_coord_f = np.linspace(lower_limit, upper_limit, y_node)
        return x_coord_f, y_coord_f
    x_coord, y_coord = mat_up_low(fem_coord, inp.shape[2], inp.shape[3])

    mode1 = SkipNet(1, 1, fem_coord, x_coord, y_coord)
    optimizer = torch.optim.Adam(mode1.parameters(), lr=0.01)
    out_dis, out_11 = mode1(inp)
    optimizer.zero_grad()
    loss = out_dis.sum()
    loss.backward()
    optimizer.step()

    # prev_loss = float('inf')
    # prev_loss = float(100.0)
    # best_model_weights = copy.deepcopy(mode1.state_dict())
    # for epoch in range(4):
    #     out_dis, out_11 = mode1(inp)
    #     optimizer.zero_grad()
    #     loss = out_dis.sum()
    #     loss.backward()
    #     optimizer.step()
    #     # print(out.size())
    #     current_loss = loss.item()
    #     print(current_loss)
        # relative_change = (current_loss - prev_loss) / current_loss
        # if relative_change > 0.01:
        #     print(f"Loss change ({relative_change:.4f}) exceeds threshold. Reverting to previous weights.")
        #     mode1.load_state_dict(best_model_weights)
        # else:
        #     print('continue')
        #     best_model_weights = copy.deepcopy(mode1.state_dict())

        # prev_loss = current_loss


