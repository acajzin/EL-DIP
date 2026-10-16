import numpy as np


def add_noise_by_snr(signal, target_snr_db):
    """
    Add Gaussian noise at the specified signal-to-noise ratio.
    :param signal:      Original signal (1D array).
    :param target_snr_db: Target signal-to-noise ratio (dB).
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def add_scale_gaussian(x, epsilon):
    """
    Add Gaussian noise of the specified strength to the signal.
    Args:
    x (array_like): Input signal (a list or NumPy array).
    epsilon (float): Noise intensity coefficient; defaults to 0.2.
    Returns:
    y (numpy.ndarray): Signal with added noise.
    """
    # Calculate the RMS value of the signal.
    rms_x = np.sqrt(np.mean(x ** 2))
    # Generate Gaussian noise.
    noise = np.random.randn(*x.shape) * epsilon * rms_x
    # Add the noise.
    y = x + noise
    return y

def noise_level(dis, noise_dis):
    n_level = np.sqrt(np.sum(np.square(dis-noise_dis)) / np.sum(np.square(dis))) * 100
    return n_level


def snr(dis, noise_dis, epsilon=1e-10):
    num = len(dis)
    signal_power = np.sum(np.square(dis)) / num
    noise_power = np.sum(np.square(dis - noise_dis)) / num
    snr_db = 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))
    return snr_db


def original_dis_xy(vtk_path):
    with open(vtk_path, 'r') as f:
        lines = f.readlines()
        dis_x_lines = lines[10581: 10581 + 2175]   # 150
        dis_y_lines = lines[12758: 12758 + 2175]
        cleaned_x_data = [float(item.strip()) for item in dis_x_lines]
        cleaned_y_data = [float(item.strip()) for item in dis_y_lines]

        # 2. Convert the lists to NumPy arrays and reshape them into 64x64 matrices.
        x_np = np.array(cleaned_x_data)
        y_np = np.array(cleaned_y_data)
    return x_np, y_np


def write_noise(noise_x, noise_y, doc_name):
    with open(doc_name, 'w') as f:
        for i in range(len(noise_x)):
            f.writelines(['1'.ljust(4, ' '), '{:e}'.format(np.around(noise_x[i], 10)).ljust(15, ' '),
                         '1'.ljust(4, ' '), '{:e}'.format(np.around(noise_y[i], 10)), '\n'])


"""
Write noisy displacements to the .in file and return the noisy x and y displacements.
"""
def noise_dis_xy(target_path, in_doc, snr_db):
    dis_x, dis_y = original_dis_xy(target_path)
    noise_x = add_noise_by_snr(dis_x, snr_db)
    noise_y = add_noise_by_snr(dis_y, snr_db)
    write_noise(noise_x, noise_y, 'noise.txt')
    with open('noise.txt', 'r') as f:
        noise_line = f.readlines()
        noise_line.append('end')
        # print(noise_line)
        print(type(noise_line))
    with open(in_doc, 'r') as f1:
        in_lines = f1.readlines()
        in_lines[8886: 8886 + 2175] = noise_line
        print(len(in_lines))
    with open(in_doc, 'w') as f2:
        f2.writelines(in_lines)
    # x_db = snr(dis_x, noise_x)
    # print(x_db)
    return noise_x, noise_y


if __name__ == "__main__":
    """
    epsilon = 20%    snr = 14db
    epsilon = 5%    snr = 26db
    epsilon = 1%    snr = 40db
    epsilon = 0.1%    snr = 60db
    """
    np.random.seed(42)
    # x=np.linspace(1,6,2000)
    # signal=np.sin(x)
    signal_x, signal_y = original_dis_xy('forward_cci0.vtk')
    # print(signal_x[:64])
    noise_signal_snr_x = add_noise_by_snr(signal_x, 60)
    print(noise_signal_snr_x[:64])

    # print(signal_y[:64])
    noise_signal_snr_y = add_noise_by_snr(signal_y, 60)
    # print(noise_signal_snr_y[:64])
    # write_noise(noise_signal_snr_x, noise_signal_snr_y, 'noise.txt')
    # db = noise_level(signal_y, noise_signal_snr_y)
    # print(db)
    # noise_signal = add_scale_gaussian(signal_y, 0.001)
    # x = snr(signal_y, noise_signal_snr_y)
    # print(x)
    x, y = noise_dis_xy('forward_cci0.vtk', 'cricle_case.in', 60)
    print(x.shape)
    print(x)
    x_db = snr(signal_x, x)
    print(x_db)
    eps = noise_level(signal_x, x)
    print(eps)
