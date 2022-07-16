# Copyright (c) OpenMMLab. All rights reserved.
# modified by ryanxingql
# Typically, there are (1+8)*2 folders to be processed for DIV2K dataset.
#     train_hq (png) -> train_hq_sub (png) -> train_hq_sub.lmdb (png)
#     train_lq/qp27 (png) -> train_lq_sub/qp27 (png) -> train_lq_sub_qp27.lmdb (png)
#     train_lq/qp32
#     train_lq/qp37
#     train_lq/qp42
#     train_lq/qf20 (jpg) -> train_lq_sub/qf20 (png) -> train_lq_sub_qf20.lmdb (png)
#     train_lq/qf30
#     train_lq/qf40
#     train_lq/qf50
#     valid_hq (png) -> valid_hq_sub.lmdb (png)
#     valid_lq/qp27 (png) -> valid_lq_sub_qp27.lmdb (png)
#     valid_lq/qp32
#     valid_lq/qp37
#     valid_lq/qp42
#     valid_lq/qf20 (jpg) -> valid_lq_sub_qf20.lmdb (png)
#     valid_lq/qf30
#     valid_lq/qf40
#     valid_lq/qf50
# Remember to modify opt configurations according to your settings.
import argparse
import os
import os.path as osp
import sys
from multiprocessing import Pool

from tqdm import tqdm
import cv2
import lmdb
import mmcv
import numpy as np


def worker(path, opt, extension_save='.png'):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is smaller
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img = mmcv.imread(path, flag='unchanged')

    if img.ndim == 2 or img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         f'{img_name}_s{index:03d}{extension_save}'),
                cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(mmcv.scandir(input_folder))
    img_list = [osp.join(input_folder, v) for v in img_list]

    prog_bar = tqdm(total=len(img_list))
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: prog_bar.update())
    pool.close()
    pool.join()
    prog_bar.close()
    print('Patching done.')


def main_extract_subimages(args, input_dir_list, save_dir_list):
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        After process, each sub_folder should have the same number of
        subimages. You can also specify scales by modifying the argument
        'scales'. Remember to modify opt configurations according to your
        settings.
    """

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size

    for input_dir, save_dir in zip(input_dir_list, save_dir_list):
        opt['input_folder'] = input_dir
        opt['save_folder'] = save_dir
        extract_subimages(opt)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(mmcv.scandir(folder_path, recursive=False)))
    keys = [img_path.split('.')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def read_img_worker(path, key, compress_level):
    """Read image worker

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """
    img = mmcv.imread(path, flag='unchanged')
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


def make_lmdb(data_path,
              lmdb_path,
              img_path_list,
              keys,
              batch=5000,
              compress_level=1,
              multiprocessing_read=False,
              n_thread=40):
    """Make lmdb.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
    """
    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        prog_bar = mmcv.ProgressBar(len(img_path_list))

        def callback(arg):
            """get the image data and update prog_bar."""
            key, dataset[key], shapes[key] = arg
            prog_bar.update()

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_path_list[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_path_list)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    prog_bar = tqdm(total=len(img_path_list))
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        prog_bar.update()
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(
                osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    txt_file.close()
    prog_bar.close()
    print('\nFinish writing lmdb.')


def make_lmdb_for_div2k(folder_paths, lmdb_paths):
    """Create lmdb files for DIV2K dataset."""

    for folder_path, lmdb_path in zip(folder_paths, lmdb_paths):
        img_path_list, keys = prepare_keys_div2k(folder_path)
        make_lmdb(folder_path, lmdb_path, img_path_list, keys)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare DIV2K dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--n-thread',
        type=int,
        default=20,
        help='thread number when using multiprocessing')
    parser.add_argument(
        '--data-root', default='./data/div2k', help='dataset root')
    parser.add_argument('--data-type', default='train', help='train or valid')
    parser.add_argument(
        '--if-hq', action='store_true', help='whether to process hq')
    parser.add_argument(
        '--if-lq', action='store_true', help='whether to process lq')
    parser.add_argument(
        '--lqs', metavar='N', type=str, nargs='+', default='qp37')
    parser.add_argument(
        '--extract-patches',
        action='store_true',
        help='whether to extract image patches')
    parser.add_argument(
        '--crop-size',
        type=int,
        default=128,
        help='cropped size for HR images')
    parser.add_argument(
        '--step', type=int, default=64, help='step size for HR images')
    parser.add_argument(
        '--thresh-size',
        type=int,
        default=0,
        help='threshold size for HR images')
    parser.add_argument(
        '--compression-level',
        type=int,
        default=3,
        help='compression level when save png images')
    parser.add_argument(
        '--make-lmdb',
        action='store_true',
        help='whether to prepare lmdb files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    input_dir_list = []
    sub_img_dir_list = []
    sub_lmdb_dir_list = []
    lmdb_dir_list = []

    if args.if_hq:
        input_dir_list.append(osp.join(args.data_root, f'{args.data_type}_hq'))
        sub_img_dir_list.append(
            osp.join(args.data_root, f'{args.data_type}_hq_sub'))
        sub_lmdb_dir_list.append(
            osp.join(args.data_root, f'{args.data_type}_hq_sub.lmdb'))
        lmdb_dir_list.append(
            osp.join(args.data_root, f'{args.data_type}_hq.lmdb'))

    if args.if_lq:
        for lq in args.lqs:
            input_dir_list.append(
                osp.join(args.data_root, f'{args.data_type}_lq/{lq}'))
            sub_img_dir_list.append(
                osp.join(args.data_root, f'{args.data_type}_lq_sub/{lq}'))
            sub_lmdb_dir_list.append(
                osp.join(args.data_root, f'{args.data_type}_lq_sub_{lq}.lmdb'))
            lmdb_dir_list.append(
                osp.join(args.data_root, f'{args.data_type}_lq_{lq}.lmdb'))

    # extract subimages
    if args.extract_patches:
        main_extract_subimages(args, input_dir_list, sub_img_dir_list)

    # prepare lmdb files
    if args.make_lmdb:
        if args.extract_patches:
            make_lmdb_for_div2k(sub_img_dir_list, sub_lmdb_dir_list)
        else:
            make_lmdb_for_div2k(input_dir_list, lmdb_dir_list)