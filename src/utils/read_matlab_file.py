import os
import os.path as osp
import glob
from pathlib import Path
import mat73
from typing import Dict, Union, List
from src.utils.get_project_root import get_project_root


def read_matlab(path: Union[Path, str, List[Union[Path, str]]]) -> List[Dict]:
    """
    :param path: path to matlab file
    :return: list of dicts
    """
    # Check if it's a single path
    if isinstance(path, (Path, str)):
        # Handle the single path case
        if isinstance(path, str):
            path = [Path(path)]

    # Handle the list of paths case
    dict_list = []
    for i, p in enumerate(path):
        if isinstance(p, (Path, str)):
            mat_dict = mat73.loadmat(p)
        else:
            raise TypeError(f'All input must be str or Path but input at index {i} was {type(p)}')
        dict_list.append(mat_dict)

    return dict_list


def test_read_matlab():
    root_path = get_project_root()
    # read example lfp file
    samples_lfp_path = [
        osp.join(root_path, 'samples', 'LFP', 'chamber_Rat4-probe3-day1-LFP_Data_Probe_A.mat'),
        osp.join(root_path, 'samples', 'LFP', 'chamber_Rat4-probe3-sniffing-day2-LFP_Data_Probe_A.mat')
    ]
    data_dict_list = read_matlab(samples_lfp_path)
    [print(d.keys()) for d in data_dict_list]


if __name__ == '__main__':
    # test_read_matlab()

    pass
