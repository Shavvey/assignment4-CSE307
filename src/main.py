import numpy as np

RED_ANSI = "\033[31m"
RESET_ANSI = "\033[0m"


def make_circle(radius: int, dims: int, thickness: int, fill_elem: int = 1):
    assert thickness < radius
    mask = np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            # (dims - 1) is to account for zero-based indexing
            x = i - (dims - 1) // 2
            y = j - (dims - 1) // 2
            p = x**2 + y**2
            if p <= radius**2 and p >= (radius - thickness) ** 2:
                mask[i][j] = fill_elem
    return mask


def print_circle(arr: np.ndarray):
    """Special print function that makes all the 1s red to better distinguish them"""
    assert len(arr.shape) == 2, "Expected some 2D numpy array as a mask"
    sb = ""
    for row in arr:
        sb += "["
        for elem in row[:-1]:
            if elem == 1:
                sb += f"{RED_ANSI}{elem: <4.1f}{RESET_ANSI}, "
            else:
                sb += f"{elem}, "
        if row[-1] == 1:
            sb += f"{RED_ANSI}{row[-1]: <4.1f}{RESET_ANSI}]\n"
        else:
            sb += f"{row[-1]}]\n"
    print(sb)


def convolve(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    assert len(mask.shape) == len(image.shape) == 2, "Expected 2D numpy arrays"
    assert (
        mask.shape[0] == mask.shape[1]
    ), "Expected square `nxn` matrix for filter/mask"
    assert image.shape[0] == image.shape[1], "Expected square `nxn` matrix for image"
    out_dims = image.shape[0] - mask.shape[0] + 1
    width = mask.shape[0]
    conv_out = np.empty((out_dims, out_dims))
    for row_idx in range(image.shape[0]):
        if row_idx + width > image.shape[0]:
            break  # check overindex
        for col_idx in range(image.shape[1]):
            # produce convolution from (i,j)
            if col_idx + width > image.shape[0]:
                # end iteration if (i,j) convolve is not possible (NOTE: we could also prob use padding instead?)
                break
            slice = image[row_idx : row_idx + width, col_idx : col_idx + width]
            # compute element wise multiplication, then sum along axis=0
            conv_out[row_idx, col_idx] = np.sum(slice * mask)
    return conv_out


def find_max_idx(input: np.ndarray) -> tuple[int, int]:
    max = -np.inf
    indices = (0, 0)
    for row_idx, row in enumerate(input):
        for col_idx, elem in enumerate(row):
            if elem > max:
                max = elem
                indices = (row_idx, col_idx)
    return indices


def matrix_ones_to_zeros(input: np.ndarray) -> np.ndarray:
    """Replace 0's inside a 2D array with 1s"""
    assert len(input.shape) == 2, f"Expecting a 2D array, was given {len(input.shape)}D"
    # this is so annoying, you could just use input[input == 0] = -1 if you wanted to use fancy indexing!
    # create array to store indices
    row_indices = np.array([], dtype=int)
    col_indices = np.array([], dtype=int)
    for row_idx, row in enumerate(input):
        for col_idx, elem in enumerate(row):
            if elem == 0:
                row_indices = np.append(row_indices, row_idx)
                col_indices = np.append(col_indices, col_idx)
    input[row_indices, col_indices] = -1
    return input


def main():
    image = make_circle(5, 28, 3)
    mask = make_circle(5, 15, 2)
    print_circle(image)
    print_circle(mask)
    conv_out_zeroes = convolve(mask, image)
    print(conv_out_zeroes)
    print(f"Max idx of zero mask convolution: {find_max_idx(conv_out_zeroes)}")
    mask = matrix_ones_to_zeros(mask)
    print_circle(mask)
    conv_out_neg_ones = convolve(mask, image)
    print(conv_out_neg_ones)
    print(
        f"Max idx of negative ones mask convolution: {find_max_idx(conv_out_neg_ones)}"
    )


if __name__ == "__main__":
    main()
