import copy
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.tri as tri
from tqdm.auto import tqdm
from matplotlib.colors import Normalize
from skip_net_cc_map import SkipNet,initialize_weights, x_y_coord_sm, idw_interp
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from in_wenjian_cc import read_sm, setup_sm, delete_doc
from test_method_cc import relative_error, min_max_normalize, sm_cc, sm_psnr, sm_ssim
from gauss_noise import noise_dis_xy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_mse(array1, array2, output_prefix="mse_calc"):
    """
    Calculate the mean squared error (MSE) between two 2D NumPy arrays.

    Args:
    array1 (np.ndarray): First 2D array (predicted values).
    array2 (np.ndarray): Second 2D array (ground-truth values).

    Returns:
    float: Mean squared error.
    """
    # # Check whether both inputs are 2D arrays.
    # if array1.ndim != 2 or array2.ndim != 2:
    #     raise ValueError("Both inputs must be 2D arrays")

    # 1. Check the array shapes.
    if array1.shape != array2.shape:
        raise ValueError("两个数组的形状必须相同")

    # 2. Calculate the difference (diff).
    diff = array1 - array2
    # Save the difference to a text file.
    np.savetxt(f"{output_prefix}_diff.txt", diff, fmt='%.8f')
    print(f"差值矩阵已保存至 {output_prefix}_diff.txt")

    # 3. Calculate the squared difference (squared_diff).
    squared_diff = np.square(diff)
    # Save the squared difference to a text file.
    np.savetxt(f"{output_prefix}_squared_diff.txt", squared_diff, fmt='%.8f')
    print(f"平方差矩阵已保存至 {output_prefix}_squared_diff.txt")

    # 4. Calculate the mean (MSE).
    mse = np.mean(squared_diff)
    
    # 5. Save the MSE result to a text file.
    with open(f"{output_prefix}_result.txt", "w") as f:
        f.write(f"Final MSE: {mse:.10f}")
    print(f"最终MSE已保存至 {output_prefix}_result.txt")

    return mse

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_(0, 1)
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)
    fill_noise(net_input, noise_type)
    net_input *= var
    return net_input


def target_dis_xy(vtk_path):
    with open(vtk_path, 'r') as f:
        lines = f.readlines()
        dis_x_lines = lines[4601: 4601 + 959]
        dis_y_lines = lines[5562: 5562 + 959]
        x_data = [float(item.strip()) for item in dis_x_lines]
        y_data = [float(item.strip()) for item in dis_y_lines]
        x_np = np.array(x_data)
        y_np = np.array(y_data)
    return x_np, y_np


def circle_view(fem_coord_c, node_values, ele_txt, p, i_th, colorbar=True, limit_range=True, is_target=False):
    nodes = fem_coord_c
    triangles = np.loadtxt(ele_txt, delimiter=",") - 1

    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles)
    # Create the figure.
    if colorbar:
        fig = plt.figure(figsize=(10, 8))
    else:
        fig = plt.figure(figsize=(8, 8))

    # Draw the filled triangles.
    if limit_range:
        tcf = plt.tripcolor(triangulation, node_values, cmap='jet', shading='gouraud', vmin=0, vmax=10)
    else:
        tcf = plt.tripcolor(triangulation, node_values, cmap='jet', shading='gouraud', vmin=0)

    # Add a colorbar.
    if colorbar:
        cbar = plt.colorbar(tcf)

    plt.axis('off')
    plt.grid(False)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    if colorbar:
        plt.savefig(p + '/sm/fig-{}-cc.png'.format(i_th), pad_inches=0.0, bbox_inches='tight')
    else:
        plt.savefig(p + '/sm/fig-{}-cc-test.png'.format(i_th), pad_inches=0.0, bbox_inches='tight')
    plt.close()


def mat_up_low(fem_coord_f, x_node, y_node):
    upper_limit = np.max(np.ceil(np.max(fem_coord_f, axis=0)))
    lower_limit = np.min(np.floor(np.min(fem_coord_f, axis=0)))
    x_coord_f = np.linspace(np.min(lower_limit), np.max(upper_limit), x_node)
    y_coord_f = np.linspace(np.min(lower_limit), np.max(upper_limit), y_node)
    return x_coord_f, y_coord_f


def read_sm(vtk_path):
    with open(vtk_path, 'r') as f:
        lines = f.readlines()
        grad_lines = lines[17112: 17112+2175]   #150
        # 1. Strip whitespace and convert the values to floats.
        cleaned_data = [float(item.strip()) for item in grad_lines]

        # 2. Convert the list to a NumPy array and reshape it into a 48x48 matrix.
        grad = np.array(cleaned_data)
        # print(grad_np)
    return grad


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def opt(loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


set_seed()
fine_coord = np.loadtxt('cc_mesh/fine_coord_150.txt', delimiter=',')
input_noise = get_noise(1, (96, 96)).to(device)
input_np = input_noise.detach().cpu().numpy()
x_coord_fine, y_coord_fine = mat_up_low(fine_coord, input_np.shape[2], input_np.shape[3])
mse = torch.nn.MSELoss()
net = SkipNet(1, 1, fine_coord, x_coord_fine, y_coord_fine).to(device)

num_iter = 2000
out_num = 100
lr_init = 0.1
optimizer = torch.optim.Adam(net.parameters(), lr_init, betas=(0.5, 0.999))

sm_target_set = setup_sm()
nn_out_coord, nn_out_sm = x_y_coord_sm(sm_target_set, x_coord_fine, y_coord_fine)
sm_ys, jb_map = idw_interp(nn_out_coord, nn_out_sm, fine_coord, p=3)      # Use p=3 for visualization.
sm_target = np.where(sm_ys > 3.0, 10.0, sm_ys)     # Obtain the target shear modulus in the circular domain.
sm_target = np.where(sm_ys < 3.0, 1.0, sm_target)

sm_target = read_sm('forward_cc_150i0.vtk')

# print(sm_target)
sm_target_norm = min_max_normalize(sm_target)

#x_target, y_target = target_dis_xy('forward_cc_150i0.vtk')
x_target, y_target = noise_dis_xy('forward_cc_150i0.vtk', 'cricle_case_150.in', snr_db=20)  # Add noise.
x_target = torch.from_numpy(x_target)
y_target = torch.from_numpy(y_target)
dis_target = torch.stack([x_target, y_target]).float().to(device)
print(dis_target.shape)
loss_list = []
# loss_list_sm = []
# loss_list_sm_norm = []
loss_lsp_list = []
loss_mse_sm_list = []
loss_cc_sm_list = []
loss_psnr_sm_list = []
loss_ssim_sm_list = []
max_sm_list = []
min_sm_list = []
path = "./cc_case5/150/20db"
prev_loss = None
lsp_num = 0
avg_num = 10
lsp_threshold = 0.1

for epoch in tqdm(range(num_iter)):
    out_dis, out_sm = net(input_noise)
    loss_total = mse(dis_target, out_dis)
    
    print(f"epoch{epoch+1}: loss={loss_total}")
    out_sm_np = out_sm.detach().cpu().numpy().squeeze()
    # sm_re = relative_error(sm_target, out_sm_np)  # Mean relative error of the shear modulus.
    # loss_list_sm.append(sm_re)
    current_loss = loss_total.item()
    # current_loss_sm = loss_sm.item()
    # print(current_loss)

    loss_list.append(current_loss)

    # Check every 10 epochs.
    if (epoch + 1) % avg_num == 0:
        avg_prev_loss = sum(loss_list[epoch-(avg_num-1):epoch+1])/avg_num
        loss_change = (current_loss - avg_prev_loss) / current_loss
        # loss_change = current_loss / avg_prev_loss
        if loss_change >= lsp_threshold:
            lsp_num += 1
            delete_doc()
            # Restore the previous parameters.
            net.load_state_dict(prev_state)    # Revert the network parameters to the previous iteration.
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            print('current_lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            out_dis_lsp, out_sm_lsp = net(input_noise)
            out_sm_np = out_sm_lsp.detach().cpu().numpy().squeeze()
            # prev_state = copy.deepcopy(net.state_dict())
            loss_lsp = mse(dis_target, out_dis_lsp)
            prev_loss = loss_lsp.item()
            opt(loss_lsp)
            # print(f'lsp_loss = {prev_loss}')
            loss_lsp_list.append(prev_loss)
        else:
            # Update the best state.
            prev_state = copy.deepcopy(net.state_dict())
            # optimizer.load_state_dict(initial_optimizer)  # lr = 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_init
            loss_lsp_list.append(current_loss)
            prev_loss = current_loss
            opt(loss_total)
    else:
        prev_state = copy.deepcopy(net.state_dict())
        opt(loss_total)
        # Initialize the reference loss.
        prev_loss = current_loss
        loss_lsp_list.append(prev_loss)
    # scheduler.step()


    if (epoch + 1) % out_num == 0:
        nn_out_coord, nn_out_sm = x_y_coord_sm(out_sm_np, x_coord_fine, y_coord_fine)
        p_sm, jb_map = idw_interp(nn_out_coord, nn_out_sm, fine_coord, p=3)  # Use p=3 for visualization.
        np.save(path + f'/sm_{epoch+1}.npy', p_sm)
        
        max_sm_list.append(np.max(p_sm))
        print(f"sm_np最大值是{np.max(p_sm)}")
        min_sm_list.append(np.min(p_sm))
        print(f"sm_np最小值是{np.min(p_sm)}")

        cc_sm = sm_cc(p_sm, sm_target)
        loss_cc_sm_list.append(cc_sm)
        # loss_mse_sm = mse(torch.tensor(sm_target), torch.tensor(p_sm))
        loss_mse_sm = calculate_mse(sm_target, p_sm)
        loss_mse_sm_list.append(loss_mse_sm)  # MSE of the shear modulus.
        psnr_value = sm_psnr(sm_target.reshape(1,-1), p_sm.reshape(1,-1))
        loss_psnr_sm_list.append(psnr_value)

        # loss_list.append(loss_total.item())
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        out_sm_np_01 = out_sm_np
        plt.imshow(out_sm_np_01, cmap='jet', vmin=0)
        plt.colorbar()
        plt.savefig(path+'/sm/fig-{}-nn.png'.format(epoch+1), pad_inches=0.1,bbox_inches='tight')
        plt.close()

        # if (epoch + 1) == num_iter:
        circle_view(fine_coord, p_sm, 'cc_mesh/fine_ele_150.txt', path, epoch + 1, colorbar=False)
        # Actual domain shape.
        # circle_view(fine_coord, p_sm, 'cc_mesh/fine_ele_150.txt', path, epoch + 1)


# Plot the displacement loss.
with open(path+'/loss/loss.txt', 'w') as f:
    for x in loss_list:
        f.writelines(['epoch{}'.format(loss_list.index(x)), '  ', str(x), '\n'])
plt.figure()
plt.title('loss')
plt.legend('loss')
plt.plot(loss_list)
plt.yscale('log')
plt.savefig(path + '/loss/fig-loss.png')

# Plot the displacement loss with LSP.
with open(path+'/loss/loss_lsp.txt', 'w') as f:
    for x in loss_lsp_list:
        f.writelines(['epoch{}'.format(loss_lsp_list.index(x)), '  ', str(x), '\n'])
plt.figure()
plt.title('loss_lsp')
plt.legend('loss_lsp')
plt.plot(loss_lsp_list)
plt.yscale('log')
plt.savefig(path + '/loss/fig-loss-lsp.png')

# sm-mes
with open(path+'/loss/loss_sm_mse.txt', 'w') as f:
    for x in loss_mse_sm_list:
        f.writelines(['epoch{}'.format(loss_mse_sm_list.index(x)), '  ', str(x), '\n'])
plt.figure()
plt.title('loss_sm_mse')
plt.legend('loss_sm_mse')
plt.plot(loss_mse_sm_list)
# plt.yscale('log')
plt.savefig(path + '/loss/fig-loss-sm-mse.png')

# sm-cc
with open(path+'/loss/loss_sm_cc.txt', 'w') as f:
    for x in loss_cc_sm_list:
        f.writelines(['epoch{}'.format(loss_cc_sm_list.index(x)), '  ', str(x), '\n'])
plt.figure()
plt.title('loss_sm_cc')
plt.legend('loss_sm_cc')
plt.plot(loss_cc_sm_list)
# plt.yscale('log')
plt.savefig(path + '/loss/fig-loss-sm-cc.png')

# Plot the target shear modulus.
circle_view(fine_coord, sm_target, 'cc_mesh/fine_ele_150.txt', path, 'target', is_target=True)
circle_view(fine_coord, sm_target, 'cc_mesh/fine_ele_150.txt', path, 'target', colorbar=False, is_target=True)
# print(sm_target_norm)

with open(path+'/loss/sm_ssim.txt', 'w') as f:
    for x in range(num_iter//out_num):
        ssim_value = sm_ssim(path+'/sm/fig-target-cc-test.png', path+'/sm/fig-{}-cc-test.png'.format((x + 1)*out_num))
        f.writelines(['ssim   epoch{}:'.format((x + 1) * out_num), '  ', str(ssim_value), '\n'])
        loss_ssim_sm_list.append(ssim_value)

plt.figure()
plt.title('loss_sm_ssim')
plt.legend('loss_sm_ssim')
plt.plot(loss_ssim_sm_list)
# plt.yscale('log')
plt.savefig(path + '/loss/fig-loss-sm-ssim.png')
plt.close()

with open(path+'/loss/loss_sm_psnr.txt', 'w') as f:
    for x in loss_psnr_sm_list:
        f.writelines(['epoch{}'.format(loss_psnr_sm_list.index(x)), '  ', str(x), '\n'])
plt.figure()
plt.title('loss_sm_pnsr')
plt.legend('loss_sm_pnsr')
plt.plot(loss_psnr_sm_list)
# plt.yscale('log')
plt.savefig(path + '/loss/fig-loss-sm-pnsr.png')
plt.close()

with open(path+'/loss/sm_max_min.txt', 'w') as f:
    for x in range(num_iter//out_num):
        f.writelines(['epoch{}:'.format((x + 1)*out_num), '  max:', str(max_sm_list[x]), '  min:', str(min_sm_list[x]),'\n'])