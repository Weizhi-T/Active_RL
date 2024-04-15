# -*- coding: utf-8 -*-
'''
@File    :   preprocess_badminton_data.py
@Time    :   2024/04/13 12:08:47
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   preprocess badminton data from video pipeline
'''

import os
import numpy as np
import pandas as pd
import pickle
import argparse


ACTIONS = {
    "drive": 0,
    "net shot": 1,
    "lob": 2,
    "clear": 3,
    "drop": 4,
    'push/rush': 5,
    "smash": 6,
    "defensive shot": 7,
    "short service": 8,
    "long service": 9
}


class DataProcesser():
    def __init__(self, dataset_path, target_file_name) -> None:
        self.dataset_path = dataset_path
        self.target_file_name = target_file_name
        self.data_path_list = []
        self.get_filepath()
    
    def get_filepath(self):
        # get all file names under directory
        if not os.path.exists(self.dataset_path):
            self.dataset_path = os.path.join(os.path.dirname(__file__), self.dataset_path)
            assert os.path.exists(self.dataset_path), "directory path not existed!"

        path_list = os.listdir(self.dataset_path)
        # print(path_list)
        for path in path_list:
            real_path = os.path.join(self.dataset_path, path)
            # print(real_path)
            if os.path.isfile(real_path):
                continue
            all_list = os.listdir(real_path)
            for file_name in all_list:
                file_path = os.path.join(real_path, file_name)
                if os.path.isfile(file_path) and "rally" in file_name:
                    self.data_path_list.append(file_path)    

    def process(self):
        data_list = []
        bad_rally_num = 0
        for data_path in self.data_path_list:
            try:
                # print(data_path)
                rally = dict()
                rally_data = pd.read_csv(data_path, converters={'ball':eval, 'top':eval, 'bottom':eval})

                # only use bottom player data
                bottom_rally_data = rally_data.loc[rally_data['pos'] == 'bottom']
                # print(bottom_rally_data.dtypes)

                # collect observations
                bottom_pos = np.array(bottom_rally_data['bottom'].tolist())
                bottom_pos = bottom_pos.reshape(bottom_pos.shape[0], -1)
                # print(bottom_pos.shape)
                top_pos = np.array(bottom_rally_data['top'].tolist())
                top_pos = top_pos.reshape(top_pos.shape[0], -1)
                # print(top_pos.shape)
                ball_pos = np.array(bottom_rally_data['ball'].tolist())
                # print(ball_pos.shape)
                # "observations": self_posture + opponent_posture + ball_position
                rally["observations"] = np.concatenate((bottom_pos, top_pos, ball_pos), axis=1)
                # print(rally['observations'].shape)

                # collect actions, one-hot encoding
                rally["actions"] = np.zeros((rally["observations"].shape[0], len(ACTIONS)))
                i = 0
                for action in bottom_rally_data['type']:
                    rally["actions"][i][ACTIONS[action]] = 1
                    i += 1
                # print(rally["actions"])

                # collect rewards
                rewards = np.zeros(rally["actions"].shape[0])
                if int(rally_data['getpoint_player'].dropna().values[0]) == bottom_rally_data['player'].values[0]:
                    rewards[-1] = 1
                else:
                    rewards[-1] = -1
                rally["rewards"] = rewards

                # collect terminals
                terminals = np.zeros(rally["actions"].shape[0])
                terminals[-1] = 1
                rally["terminals"] = terminals

                data_list.append(rally)
            except:
                bad_rally_num += 1
                print("ignore bad data.")

        print("==== total episodes/rallies: {} ===".format(len(data_list)))
        print("==== bad episodes/rallies: {} ===".format(bad_rally_num))
        target_path = os.path.join(os.path.dirname(__file__), self.target_file_name)
        with open(f'{target_path}.pkl', 'wb') as f:
            pickle.dump(data_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data4drl")
    parser.add_argument("--target_file_name", type=str, default="badminton_test")

    args = parser.parse_args()
    
    processer = DataProcesser(args.dataset_path, args.target_file_name)
    processer.process()