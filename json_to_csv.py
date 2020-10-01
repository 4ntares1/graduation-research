import json
import pandas as pd
import numpy as np
import os
import sys


FILE_PATH = 'csv'


# DataFrameのフォーマット
class DataFrameFormat:
    _columns_list = ['X-axis', 'Y-axis', 'Confidence']
    _part_list = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
                  'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
                  'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                  'REye', 'LEye', 'REar', 'LEar', 'LBigToe',
                  'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
    _part_qty = len(_part_list)


# 読み込むファイルの情報を入力する関数
def input_file_data():
    # 変換したいデータがいくつあるのかの入力を求める
    print('How many data?')
    data_qty = int(input().strip())
    # 保存用リスト作成
    data_name_list = []
    for i in range(data_qty):
        # 元動画の名前（jsonファイルの頭の部分）の入力を求める
        print('\nPlese name the data' + str(i + 1))
        data_name_list.append((input().strip()))
    return data_name_list


# 読み込んだjsonファイルの値をリストに保存する関数
def convert_json_to_list(data_name):
    # カウンタ作成
    file_count = 0
    people_qty = 0
    # jsonファイル読み込み
    while os.path.isfile(data_name + '_{:0>12}_keypoints.json'.format(
            file_count)):
        with open(data_name + '_{:0>12}_keypoints.json'.format(
                file_count)) as f:
            json_data = json.load(f)
            if file_count == 0:
                # 保存用リスト作成
                x_axis = [[[] for i in range(DataFrameFormat._part_qty)
                           ] for j in range(len(json_data['people']))]
                y_axis = [[[] for i in range(DataFrameFormat._part_qty)
                           ] for j in range(len(json_data['people']))]
                confidence = [[[] for i in range(DataFrameFormat._part_qty)
                               ] for j in range(len(json_data['people']))]
            # DataFrameの情報をリストに保存する
            for people_count, p in enumerate(json_data['people']):
                json_data = np.array(p['pose_keypoints_2d']).reshape(-1, 3)
                df_json = pd.DataFrame(json_data,
                                       columns=DataFrameFormat._columns_list,
                                       index=DataFrameFormat._part_list)
                for part_count in range(DataFrameFormat._part_qty):
                    x_axis[people_count][part_count].append(float(
                        df_json.at[DataFrameFormat._part_list[part_count],
                                   DataFrameFormat._columns_list[0]]))
                    y_axis[people_count][part_count].append(float(
                        df_json.at[DataFrameFormat._part_list[part_count],
                                   DataFrameFormat._columns_list[1]]))
                    confidence[people_count][part_count].append(float(
                        df_json.at[DataFrameFormat._part_list[part_count],
                                   DataFrameFormat._columns_list[2]]))
                # 人数保存
                if people_qty < people_count:
                    people_qty = people_count
        file_count += 1
    people_qty += 1
    if file_count == 0:
        sys.exit('error: ' + data_name + ' is No data')
    return people_qty, x_axis, y_axis, confidence


# フォルダがなければ作る関数
def make_folder(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        print('complete make folder: ' + folder_name)


# リストの値をDataFrameにしてcsvファイルに書き込む関数
def convert_list_to_csv(data_name, people_qty, x_axis, y_axis, confidence):
    # csvを保存するフォルダ作成
    folder_path = make_folder_path(data_name)
    make_folder(folder_path)
    for people_count in range(people_qty):
        for part_count in range(DataFrameFormat._part_qty):
            csv_file_name = make_csv_file_name(data_name,
                                               people_count + 1,
                                               part_count)
            # リストの値をDataFrameにする
            df = pd.DataFrame({DataFrameFormat._columns_list[0]: x_axis[
                people_count][part_count],
                               DataFrameFormat._columns_list[1]: y_axis[
                                   people_count][part_count],
                               DataFrameFormat._columns_list[2]: confidence[
                                   people_count][part_count]})
            # DataFrameのデータをcsvに保存
            df.to_csv(folder_path + '/' + csv_file_name + '.csv')
            print('complete make file: ' + csv_file_name + '.csv')
            '''
            # csvファイルにヘッダ付与
            with open(csv_file_path + '/' + csv_file_name + '.csv', "r") as f:
                original_data = f.read()
            with open(csv_file_path + '/' + csv_file_name + '.csv', 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[csv_file_name])
                writer.writeheader()
                f.write(original_data)
            '''


def make_folder_path(data_name):
    folder_path = FILE_PATH + '/' + data_name
    return folder_path


# ファイル名を作成する
def make_csv_file_name(data_name, people_number, part_number):
    # ファイル名入力
    file_name = data_name
    file_name += '(person' + str(people_number) + '-part[' + str(
        part_number) + ',' + DataFrameFormat._part_list[part_number] + '])'
    return file_name


# main
def main():
    # 読み込むデータの数、ファイルの名前、開始ファイル、終了ファイルの情報を取得
    data_name_list = input_file_data()
    # 保存用フォルダ作成
    make_folder(FILE_PATH)
    for i in range(len(data_name_list)):
        # jsonファイルの情報をリストに
        people_qty, x_axis, y_axis, confidence = convert_json_to_list(
            data_name_list[i])
        # リストの値をcsvファイルに
        convert_list_to_csv(data_name_list[i], people_qty,
                            x_axis, y_axis, confidence)


if __name__ == '__main__':
    main()