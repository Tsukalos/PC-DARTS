import os
import numpy as np
import jpeg
import random
from torch.utils.data import Dataset
import torch
from typing import List
from math import floor

ZIG_ZAG_ORDER = [0,  1,  5,  6, 14, 15, 27, 28,
                 2,  4,  7, 13, 16, 26, 29, 42,
                 3,  8, 12, 17, 25, 30, 41, 43,
                 9, 11, 18, 24, 31, 40, 44, 53,
                 10, 19, 23, 32, 39, 45, 52, 54,
                 20, 22, 33, 38, 46, 51, 55, 60,
                 21, 34, 37, 47, 50, 56, 59, 61,
                 35, 36, 48, 49, 57, 58, 62, 63]

UCF_SIZE = (320, 240)

def np_norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def get_num_files(dir_path: str):
    return len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

def bres(x1,y1,x2,y2):
    x,y = x1,y1
    dx = abs(x2 - x1)
    dy = abs(y2 -y1)
    if dx == 0:
        dx = 0.00001
    gradient = dy/float(dx)
    if dx == 0.00001:
        dx = 0


    if gradient > 1:
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2*dy - dx
    xcoordinates = [x]
    ycoordinates = [y]

    for k in range(2, dx + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

        x = x + 1 if x < x2 else x - 1

        xcoordinates.append(x)
        ycoordinates.append(y)
    return xcoordinates, ycoordinates

class UCF101DCTDataset(Dataset):

    def __init__(self, 
                 root,
                dataset_video_list_path: str,
                splits_path: str, 
                transform=None,
                fold: int = 1,
                train: bool = True,
                num_coeff: int = 32,
                seed: int = 2,
                shuffle: bool = True) -> None:
        super().__init__()

        self.root = root
        self.dataset_path = dataset_video_list_path
        self.transform = transform
        self.train = train
        self.fold = fold
        self.dct_coeffs = num_coeff
        self.seed = seed
        self.annotation_path = splits_path

        self.video_list = []

        with open(self.dataset_path, "r") as f:
            for line in f:
                a = line.split()
                self.video_list.append((a[0], a[1]))
        indices = self._select_fold(
            self.video_list, self.annotation_path, self.fold, self.train)

        random.seed(self.seed)

        if shuffle:
            random.shuffle(indices)

        # indices = random.sample(indices, 512)
        self.selected_videos = [self.video_list[i] for i in indices]
        self.frames_of_folder = []
        for v, _ in self.selected_videos:
            self.frames_of_folder.append(get_num_files(v))

    def __len__(self) -> int:
        return len(self.selected_videos)

    def __getitem__(self, index):

        num_frames = 15
        if self.train:
            num_frames = 3

        vid_frames_folder = self.selected_videos[index]

        folder = vid_frames_folder[0]
        label = vid_frames_folder[1]

        frames_list = []

        num_files = self.frames_of_folder[index]

        if num_files < num_frames:
            selected_frames = list(range(num_files))
            v = selected_frames[-1]
            for _ in range(num_frames-num_files):
                selected_frames.append(v)

        elif num_files > num_frames:
            selected_frames = list(
                range(0, num_files, floor(num_files/num_frames)))
            selected_frames = random.sample(selected_frames, num_frames)
            selected_frames.sort()
        else:
            selected_frames = list(range(num_files))
        num_selected = len(selected_frames)
        if num_selected != num_frames:
            raise ValueError("Wrong number of frames selected for parsing.")

        for i in selected_frames:
            frame_path = os.path.join(folder, f"frame_{i+1:04d}.jpg")

            parsed_frame = jpeg.parse(frame_path)
            parsed_frame = parsed_frame.transpose(3, 1, 2, 0)
            parsed_frame = parsed_frame[ZIG_ZAG_ORDER[0:self.dct_coeffs]]

            if self.transform is not None:
                parsed_frame = self.transform(parsed_frame)

            frames_list.append(parsed_frame)

        frames_list = torch.stack(frames_list)
        # frames_list = (frames_list-torch.min(frames_list))/(torch.max(frames_list)-torch.min(frames_list))
        return frames_list, torch.tensor(int(label)-1)

    def _select_fold(self, video_list: List[str], annotation_path: str, fold: int, train: bool) -> List[int]:
        name = "train" if train else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(self.root, *x.split("/")) for x in data]
            selected_files.update(data)
        indices = [i for i in range(len(video_list))
                   if video_list[i][0] in selected_files]
        return indices


class UCF101DCTDataset_MV(Dataset):

    def __init__(self, 
                 root,
                dataset_video_list_path: str,
                splits_path: str,
                mv_list_path: str, 
                transform=None,
                mv_transform=None,
                fold: int = 1,
                train: bool = True,
                num_coeff: int = 32,
                seed: int = 2,
                shuffle: bool = True,
                normalize_I_frames: bool = False):
        
        super().__init__()

        self.root = root
        self.dataset_path = dataset_video_list_path
        self.mv_path = mv_list_path
        self.transform = transform
        self.mv_transform = mv_transform
        self.train = train
        self.fold = fold
        self.dct_coeffs = num_coeff
        self.seed = seed
        self.annotation_path = splits_path
        self.normalize = normalize_I_frames

        self.video_list = []
        self.mv_list = []

        with open(self.dataset_path, 'r') as f:
            for line in f:
                a = line.split()
                self.video_list.append((a[0], a[1]))
        with open(self.mv_path, 'r') as f:
            for line in f:
                a = line.split()
                self.mv_list.append((a[0], a[1]))
        
        indices = self._select_fold(
            self.video_list, self.annotation_path, self.fold, self.train)

        random.seed(self.seed)

        if shuffle:
            random.shuffle(indices)

        # indices = random.sample(indices, 512)
        self.selected_videos = [self.video_list[i] for i in indices]
        self.selected_mvs = [self.mv_list[i] for i in indices]
        self.frames_of_folder = []
        for v, _ in self.selected_videos:
            self.frames_of_folder.append(get_num_files(v))

    def __len__(self) -> int:

        return len(self.selected_videos)


####### old get item 
    # def __getitem__(self, index):

    #     num_frames = 10
    #     if self.train:
    #         num_frames = 3

    #     vid_frames_folder = self.selected_videos[index]
    #     mvs_folder = self.selected_mvs[index]

    #     video_folder = vid_frames_folder[0]
    #     mv_folder = mvs_folder[0]
    #     label = vid_frames_folder[1]

    #     frames_list = []
    #     mv_list = []

    #     num_files = self.frames_of_folder[index]

    #     if num_files < num_frames:
    #         selected_frames = list(range(num_files))
    #         v = selected_frames[-1]
    #         for _ in range(num_frames-num_files):
    #             selected_frames.append(v)

    #     elif num_files > num_frames:
    #         selected_frames = list(range(0, num_files, floor(num_files/num_frames)))
    #         selected_frames = random.sample(selected_frames, num_frames)
    #         selected_frames.sort()
    #     else:
    #         selected_frames = list(range(num_files))

    #     num_selected = len(selected_frames)
    #     if num_selected != num_frames:
    #         raise ValueError("Wrong number of frames selected for parsing.")

    #     for i in selected_frames:
    #         frame_path = os.path.join(video_folder, f"frame_{i+1:04d}.jpg")
    #         mv_path = os.path.join(mv_folder, f"frame_{i+1:04d}.npy")

    #         frame_mv = self._load_mv(mv_path)
    #         parsed_frame = jpeg.parse(frame_path)

    #         parsed_frame = parsed_frame.transpose(3, 1, 2, 0)
    #         if self.normalize:
    #             parsed_frame = np_norm(parsed_frame)
    #         parsed_frame = parsed_frame[ZIG_ZAG_ORDER[0:self.dct_coeffs]]
            
    #         if self.transform is not None:
    #             parsed_frame = self.transform(parsed_frame)

    #         if self.mv_transform is not None:
    #             frame_mv = self.mv_transform(frame_mv)

    #         frames_list.append(parsed_frame)
    #         mv_list.append(frame_mv)

    #     frames_list = torch.stack(frames_list)
    #     mv_list = torch.stack(mv_list)
    #     return frames_list, mv_list, torch.tensor(int(label)-1)
    

    def __getitem__(self, index):

        num_frames = 10
        if self.train:
            num_frames = 3

        vid_frames_folder = self.selected_videos[index]
        mvs_folder = self.selected_mvs[index]

        video_folder = vid_frames_folder[0]
        mv_folder = mvs_folder[0]
        label = vid_frames_folder[1]

        frames_list = []
        mv_list = []
        selected_frames = []
        
        num_files = self.frames_of_folder[index]

        frame_diff = num_files - num_frames

        if frame_diff < 0:
            selected_frames = list(range(num_files))
            v = selected_frames[-1]
            for _ in range(num_frames-num_files):
                selected_frames.append(v)
        elif frame_diff == 0:
            selected_frames = list(range(num_files))
        else:
            start_point = random.randint(0, frame_diff)
            selected_frames = list(range(start_point,start_point+num_frames))

        num_selected = len(selected_frames)
        if num_selected != num_frames:
            raise ValueError("Wrong number of frames selected for parsing.")

        for i in selected_frames:
            frame_path = os.path.join(video_folder, f"frame_{i+1:04d}.jpg")
            mv_path = os.path.join(mv_folder, f"frame_{i+1:04d}.npy")

            frame_mv = self._load_mv(mv_path)
            parsed_frame = jpeg.parse(frame_path)

            parsed_frame = parsed_frame.transpose(3, 1, 2, 0)
            if self.normalize:
                parsed_frame = np_norm(parsed_frame)
            parsed_frame = parsed_frame[ZIG_ZAG_ORDER[0:self.dct_coeffs]]
            
            if self.transform is not None:
                parsed_frame = self.transform(parsed_frame)

            if self.mv_transform is not None:
                frame_mv = self.mv_transform(frame_mv)

            frames_list.append(parsed_frame)
            mv_list.append(frame_mv)

        frames_list = torch.stack(frames_list)
        mv_list = torch.stack(mv_list)
        return frames_list, mv_list, torch.tensor(int(label)-1)

    def _select_fold(self, video_list: List[str], annotation_path: str, fold: int, train: bool) -> List[int]:

        name = "train" if train else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(self.root, *x.split("/")) for x in data]
            selected_files.update(data)
        indices = [i for i in range(len(video_list))
                   if video_list[i][0] in selected_files]
        return indices
    
    def _load_mv(self, path: str):

        mvs = np.load(path)
        arr = np.zeros((int(UCF_SIZE[0]/8),int(UCF_SIZE[1]/8)), dtype=np.uint8)
        for mv in mvs:
            x1, y1 = mv[3], mv[4]
            x2, y2 = mv[5], mv[6]
            cx, cy = bres(x1,y1,x2,y2)
            for x, y in zip(cx, cy):
                if x >= UCF_SIZE[0] or y >= UCF_SIZE[1]:
                    continue
                arr[int(x/8)-1][int(y/8)-1] += 1

        if np.max(arr) == 1:
            arr[:] = 0
        return arr.transpose(1,0)[:,:, np.newaxis]