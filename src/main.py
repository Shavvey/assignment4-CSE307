import numpy as np

RED_ANSI = "\033[31m"
RESET_ANSI = "\033[0m"


def make_mask(radius: int, dims: int, thickness: int):
    assert thickness < radius
    mask = np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            x = i - (dims - 1) // 2
            y = j - (dims - 1) // 2
            p = x**2 + y**2
            if p <= radius**2 and p >= (radius - thickness) ** 2:
                mask[i][j] = 1
    return mask


def print_mask(arr: np.ndarray):
    """Special print function that makes all the 1s red to better distinguish them"""
    assert len(arr.shape) == 2, "Expected some 2D numpy array as a mask"
    sb = ""
    for row in arr:
        sb += "["
        for elem in row[:-1]:
            if elem == 1:
                sb += f"{RED_ANSI}{elem}{RESET_ANSI}, "
            else:
                sb += f"{elem}, "
        if row[-1] == 1:
            sb += f"{RED_ANSI}{row[-1]}{RESET_ANSI}\n"
        else:
            sb += f"{row[-1]}\n"
    print(sb)


def main():
    mask = make_mask(5, 28, 3)
    print_mask(mask)


if __name__ == "__main__":
    main()
