import numpy as np

def load_data(base_path_to_data: str, seed: int) -> np.lib.npyio.NpzFile:
    """
    Load a dataset from a specified base path and seed.

    Parameters:
    base_path_to_data (str): The base path to the directory containing the dataset.
    seed (int): The seed used to generate the dataset.

    Returns:
    np.lib.npyio.NpzFile: The loaded dataset as an .npz object.
    """
    dataset_path = f'{base_path_to_data}/data/1d_gaussians_{seed}.npz'
    dataset = np.load(dataset_path, allow_pickle=True)
    return dataset