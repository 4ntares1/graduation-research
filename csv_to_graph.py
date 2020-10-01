import os
import sys
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
# サーバ等のローカル以外で実行する場合は下の行を有効にする('Agg')の部分はファイル保存形式による
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import axes3d, Axes3D
import json_to_csv as jtoc
from json_to_csv import DataFrameFormat

# 使用部位番号
ANGLE_PAIR_LIST = [[[9, 10, 11], [12, 13, 14]]]
# , [[10, 11, 22], [13, 14, 19]]
# confidenceがこの値より低いなら処理をする閾値
MINIMUM_PROBABILITY = 0.4
# MINIMUM_PROBABILITY未満の累積度数分布がこの値以下ならデータを破棄する閾値
DELETION_JUDGMENT_PROBABILITY = 0.5
# 補間方法
KIND = 'linear'
# 移動平均間隔
KERNEL_SIZE = 3
# 動作開始終了判定閾値
START_AND_STOP_THRESHOLD = 0.2
# 結果のファイルパス
RESULT_FOLDER_PATH = '結果'
GRAPH_PATH = 'グラフ'
MIN_ANGLE_FILE_PATH = '角度の最小値'
LOW_ANGLE_FILE_PATH = '角度の極小値の中央値'
T_FILE_PATH = '歩行周期の中央値'
# 動画のfps
FPS = 30
# グラフ横軸名称
GRAPH_X_AXIS_NAME = '時間[s]'
GRAPH_Y_AXIS_NAME = '関節角度[ °]'


# 座標データ
class CoordinatePointsData:
    def __init__(self, x_axis, y_axis, confidence):
        self.c = confidence
        point = [Parse.smoothing(Parse.interpolation(x_axis, self.c)),
                 Parse.smoothing(Parse.interpolation(y_axis, self.c))]
        self.p = np.array(point)


# 角度データ
class AngleData:
    def __init__(self, angle):
        self.a = angle
        self.ga = np.gradient(self.a)

        self.start, self.stop = Parse.serch_start_and_stop(
            Parse.normalization(abs(self.ga)))

    def cut_off_start_end(self, graph_start, graph_stop):
        self.ca = Parse.low_pass(self.a, graph_start, graph_stop)
        self.cga = Parse.cut_off(Parse.normalization(abs(self.ga)),
                                 graph_start, graph_stop)

        self.min_ca = min(self.ca)
        peak_index = Parse.calculate_extreme_value(self.ca)
        self.high_peak_index = peak_index[0]
        self.low_peak_index = peak_index[1]
        self.low_ca = Parse.calculate_median_low_peak(self.ca)
        self.T = Parse.calculate_period(self.high_peak_index)


class Parse:
    # 補間
    def interpolation(function, c):
        # 検出信頼値が特定の数値以上のデータのインデックス取得
        index_list = [i for i, value in enumerate(
                c) if value >= MINIMUM_PROBABILITY]
        interpolation_function = np.zeros(function.shape)
        if len(index_list) >= 2:
            index = np.zeros(1)
            data = np.array(function[0])
            # 検出信頼値が特定の数値以上のデータ取得
            for i in range(1, len(function) - 1):
                if i in index_list:
                    index = np.append(index, i)
                    data = np.append(data, function[i])
            index = np.append(index, len(function) - 1)
            data = np.append(data, function[-1])
            # 検出信頼値が特定の数値以上のデータを補間
            temp_function = interp1d(index, data, kind=KIND,
                                     bounds_error=False, fill_value=0.0)
            for i in range(len(function)):
                interpolation_function[i] = temp_function(i)
        return interpolation_function

    # 平滑化(移動平均)
    def smoothing(function):
        v = np.ones(KERNEL_SIZE) / KERNEL_SIZE
        temp = np.convolve(function, v, mode='valid')
        smoothing_function = np.delete(temp,
                                       np.s_[len(temp) -
                                             (KERNEL_SIZE - 1) // 2:])
        return smoothing_function

    # 正規化
    def normalization(x, axis=None):
        x_min = x.min(axis=axis, keepdims=True)
        x_max = x.max(axis=axis, keepdims=True)
        return (x - x_min) / (x_max - x_min)

    # 特定の値以上になった最初と、特定の値以下になった最後のインデックスを取得
    def serch_start_and_stop(function):
        function_list = [d for d in function]
        temp = [i for i, x in enumerate(
            function_list) if x > START_AND_STOP_THRESHOLD]
        if not (temp == [] or len(temp) == 1):
            start_index = temp[0]
            stop_index = temp[-1]
        else:
            start_index = 0
            stop_index = len(function_list) - 1
        return start_index, stop_index

    # 左と右の部位でグラフの開始と終了を決定する
    def calculate_start_and_stop(start1, stop1, start2, stop2):
        start = min(start1, start2) - FPS // 2
        stop = max(stop1, stop2) + FPS // 2
        return start, stop

    # 開始地点と終了地点からデータを切り取る
    def cut_off(function, start, stop):
        if start < 0:
            start = 0
        if stop > len(function) - 1:
            stop = len(function) - 1
        temp = np.delete(function, np.s_[stop:])
        return np.delete(temp, np.s_[: start])

    # 極値のインデックス取得
    def calculate_extreme_value(function):
        high_peak_index = []
        low_peak_index = []
        threshod = (max(function) + min(function)) / 2
        # 極大値のインデックス取得
        high_array = np.where(function > threshod)[0]
        high_list = Parse.compact_number_list(high_array.tolist())
        for high in high_list:
            high_peak_index.append(
                min(high) + np.argmax(function[min(high): max(high) + 1]))
        # 極小値のインデックス取得
        low_array = np.where(function < threshod)[0]
        low_list = Parse.compact_number_list(low_array.tolist())
        for low in low_list:
            low_peak_index.append(
                min(low) + np.argmin(function[min(low): max(low) + 1]))
        return high_peak_index, low_peak_index

    # 連番判定
    def compact_number_list(lst):
        if 1 < len(lst):
            seq = []
            [seq.append([e]) if 0 <= i and e != lst[i - 1] + 1
             else seq[-1].append(e) for i, e in enumerate(lst)]
            return map(lambda x: [x[0], x[-1]] if len(x) > 1 else [x[0]], seq)
        return lst

    # 極小値の中央値計算
    def calculate_median_low_peak(function):
        low_peak = []
        threshod = (max(function) + min(function)) / 2
        # 極小値のインデックス取得
        low_array = np.where(function < threshod)[0]
        low_list = Parse.compact_number_list(low_array.tolist())
        for low in low_list:
            low_peak.append(min(function[min(low): max(low) + 1]))
        return np.median(np.array(low_peak))

    # 周期(極値のインデックス間隔の中央値)計算
    def calculate_period(index_list):
        temp = 0
        interval_list = []
        for i, index in enumerate(index_list):
            if not i == 0:
                interval_list.append(index - temp)
            temp = index
        return np.median(np.array(interval_list)) / FPS

    # 平滑化
    def low_pass(function, start, stop):
        temp = np.delete(function, np.s_[: start])
        # f = np.delete(temp, np.s_[stop - start:])
        N = len(function)
        fc = 2 # Parse.calculate_ff1(f) + 2
        fn = FPS / 2
        freq = np.linspace(0, FPS, N)
        F = np.fft.fft(function) / (N / 2)
        F[0] = F[0] / 2
        G = F.copy()
        G[((freq > fc) | (freq >= fn) | (freq < 0))] = 0
        g = np.fft.ifft(G)
        temp = np.real(g * N)
        temp = np.delete(temp, np.s_[: start + 1])
        smoothing_function = np.delete(temp, np.s_[stop - start:])
        '''
        dt = 1 / FPS
        t = np.arange(0, N * dt, dt)
        fig = plt.figure()
        moto = fig.add_subplot(2, 2, 1)
        moto_f = fig.add_subplot(2, 2, 2)
        ato = fig.add_subplot(2, 2, 3)
        ato_f = fig.add_subplot(2, 2, 4)
        moto.plot(t, function)
        moto_f.plot(freq, F)
        ato.plot(t, np.real(g * N))
        ato_f.plot(freq, G)
        plt.show
        '''
        return smoothing_function

    # 基本周波数計算
    def calculate_ff1(function):
        # sf = moving_average(function)
        N = len(function)
        F = np.fft.fft(function)
        ff = (np.argmax(F[1:]) + 1) / N / FPS
        return ff


# 読み込むファイルの情報を入力する
def input_file_data():
    # グラフ化したいデータがいくつあるのかの入力を求める
    print('\nHow many data?')
    data_qty = int(input().strip())

    # 保存用リスト作成
    data_name_list = []
    person_number_list = []
    for i in range(data_qty):
        # 元動画の名前（jsonファイルの頭の部分）の入力を求める
        print('\nPlese name the data' + str(i + 1))
        data_name_list.append((input().strip()))
        # グラフ化したい人物が何人目なのかの入力を求める
        person_number_list.append(1)
        '''
        while 1:
            print('\nWhich peple number?(1~100)')
            person_number = int(input().strip())
            if person_number >= 1 and person_number <= 100:
                person_number_list.append(person_number)
                break
        '''
    print('Plese wait...')
    return data_name_list, person_number_list


# 結果保存用フォルダの名前作成
def make_folder_path(data_name_list, person_number_list):
    folder_name_list = []
    for i, data_name in enumerate(data_name_list):
        folder_name_list.append(
            data_name + '(' + str(person_number_list[i]) + ')')
    folder_path = RESULT_FOLDER_PATH + '/'
    folder_path += str(folder_name_list)
    return folder_path


# csvファイル検索して、なかった場合は作成する
def serch_csv(data_name_list, person_number_list, part_number):
    file_name_list = []
    for i, data_name in enumerate(data_name_list):
        file_name_list.append(jtoc.make_folder_path(data_name) + '/' +
                              jtoc.make_csv_file_name(data_name,
                                                      person_number_list[i],
                                                      part_number) +
                              '.csv')
        if not os.path.isfile(file_name_list[i]):
            # jsonファイルの情報をリストに
            person_qty, x_axis, y_axis, confidence = jtoc.convert_json_to_list(
                data_name)
            # csvを保存するフォルダ作成
            jtoc.make_folder(data_name)
            # リストの値をcsvファイルに
            jtoc.convert_list_to_csv(data_name, person_qty,
                                     x_axis, y_axis, confidence)
            if not os.path.isfile(file_name_list[i]):
                sys.exit('error: No data exists for that person')
    return file_name_list


# csvファイルからリストに値を読み込む
def read_csv(file_name_list):
    data_list = []
    for file_name in file_name_list:
        data = pd.read_csv(file_name, encoding='utf-8')
        x_axis = np.array(data[DataFrameFormat._columns_list[0]])
        y_axis = np.array(data[DataFrameFormat._columns_list[1]])
        confidence = np.array(data[DataFrameFormat._columns_list[2]])
        data_list.append(CoordinatePointsData(x_axis, y_axis, confidence))
    return data_list


# 二次元リスト行列入れ替え(転置)
def organize_by_data(data_list_list):
    return [list(x) for x in zip(*data_list_list)]


# 削除判定(部位)
def deletion_judgment1(part_list, data_name, person_number):
    flag_list1 = []
    flag_list2 = []
    for i, data in enumerate(part_list):
        ch = np.histogram(data.c * 100, range=(0, 100))
        relative = ch[0] / np.sum(ch[0])
        cumulative = np.zeros(relative.shape)
        for j, re in enumerate(relative):
            if j == 0:
                cumulative[j] = re
            else:
                cumulative[j] = cumulative[j - 1] + re
        flag_list1.append(cumulative[int(MINIMUM_PROBABILITY * 10) - 1] <
                          DELETION_JUDGMENT_PROBABILITY)
    for angle_list in ANGLE_PAIR_LIST:
        temp = []
        for i in range(0, len(angle_list) * 3, 3):
            temp.append(
                all([flag_list1[i], flag_list1[i + 1], flag_list1[i + 2]]))
        flag_list2.append(all(temp))
    for i, flag in enumerate(flag_list2):
        if not flag:
            print('delete data:' + make_legend_list(data_name, person_number) +
                  make_graph_name(i * 2))
    return flag_list2


# 削除判定(角度)
def deletion_judgment2(pair_angle_flag_list):
    flag_list = []
    for i, data_flag in enumerate(pair_angle_flag_list):
        flag = all(data_flag)
        if not flag:
            print('delete data:' + make_graph_name(i * 2))
        flag_list.append(flag)
    return flag_list


# 角度計算
def calculate_angle(part_list):
    angle = []
    for i, angle_list in enumerate(ANGLE_PAIR_LIST):
        for j in range(0, len(angle_list) * 3, 3):
            angle.append(AngleData(joint_angle([part_list[i * 6 + j],
                                                part_list[i * 6 + j + 1],
                                                part_list[i * 6 + j + 2]])))
    return angle


# 角度計算
def joint_angle(part_list):
    p_x = []
    p_y = []
    a = []
    for part in part_list:
        p_x.append(part.p[0])
        p_y.append(part.p[1])
    u = np.array([p_x[0] - p_x[1], p_y[0] - p_y[1]])
    v = np.array([p_x[2] - p_x[1], p_y[2] - p_y[1]])
    for j in range(u.shape[1]):
        i = np.inner(u[:, j], v[:, j])
        n = np.linalg.norm(u[:, j]) * np.linalg.norm(v[:, j])
        if n == 0:
            a.append(0)
        else:
            c = i / n
            a.append(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.array(a)


# 動作開始地点と終了地点を入力
def serch_start_and_stop(angle_list):
    start_and_stop_list = []
    for i in range(0, len(angle_list), 2):
        start, stop = Parse.calculate_start_and_stop(angle_list[i].start,
                                                     angle_list[i].stop,
                                                     angle_list[i + 1].start,
                                                     angle_list[i + 1].stop)
        start_and_stop_list.append([start, stop])
        start_and_stop_list.append([start, stop])
    return start_and_stop_list


# グラフ作成
def make_graph(angle_list, name_list, person_list, folder_path, flag):
    for angle_number in range(len(angle_list)):
        if angle_number % 2 == 0 and flag[angle_number // 2]:
            # グラフの横軸にプロットするデータ取得
            frame_list1 = make_graph_frame_list(angle_list[angle_number])
            frame_list2 = make_graph_frame_list(angle_list[angle_number + 1])
            # グラフの横軸の目盛りの上限取得
            frame_qty1 = make_graph_frame_qty(frame_list1)
            frame_qty2 = make_graph_frame_qty(frame_list2)
            # グラフの線の名前作成
            legend_list = make_legend_list(name_list, person_list)
            # フォント設定
            fp = FontProperties(fname=r'C:\WINDOWS\Fonts\msgothic.ttc',
                                size=24)
            # グラフ名作成
            graph_name = make_graph_name(angle_number)
            a1_name = '右膝'
            a2_name = '左膝'
            # グラフフォーマット
            fig = plt.figure(figsize=(17, 10), dpi=100)
            plt.subplots_adjust(hspace=0.4)
            a1 = fig.add_subplot(2, 1, 1)
            a2 = fig.add_subplot(2, 1, 2)
            a1.set_title(a1_name, fontname='MS Gothic', fontsize=24)
            a2.set_title(a2_name, fontname='MS Gothic', fontsize=24)
            # 軸のラベル設定（横軸）
            a1.set_xlabel(GRAPH_X_AXIS_NAME, fontname='MS Gothic', fontsize=24)
            a2.set_xlabel(GRAPH_X_AXIS_NAME, fontname='MS Gothic', fontsize=24)
            # 軸のラベル設定（縦軸）
            a1.set_ylabel(GRAPH_Y_AXIS_NAME, fontname='MS Gothic', fontsize=24)
            a2.set_ylabel(GRAPH_Y_AXIS_NAME, fontname='MS Gothic', fontsize=24)
            # 目盛り設定（横軸）
            a1.set_xticks(np.arange(0, frame_qty1, 1))
            a1.set_xticks(np.arange(0, frame_qty1, 1 / (FPS / 10)),
                          minor=True)
            a1.tick_params(labelsize=24, width=2, length=6)
            a2.set_xticks(np.arange(0, frame_qty2, 1))
            a2.set_xticks(np.arange(0, frame_qty2, 1 / (FPS / 10)),
                          minor=True)
            a2.tick_params(labelsize=24, width=2, length=6)
            # 目盛り設定（縦軸）
            a1.set_yticks(np.arange(0, 370, 10))
            a2.set_yticks(np.arange(0, 370, 10))
            # データプロット
            for i, angle in enumerate(angle_list[angle_number]):
                if i == 1:
                    a1.plot(frame_list1[i], angle.ca, lw=2, color='black',
                            linestyle='dashed')
                else:
                    a1.plot(frame_list1[i], angle.ca, lw=2, color='black',
                            linestyle='solid')
            for i, angle in enumerate(angle_list[angle_number + 1]):
                if i == 1:
                    a2.plot(frame_list2[i], angle.ca, lw=2, color='black',
                            linestyle='dashed')
                else:
                    a2.plot(frame_list2[i], angle.ca, lw=2, color='black',
                            linestyle='solid')
            # 線の名前を表示
            a1.legend(legend_list, prop=fp, loc='lower right',
                      bbox_to_anchor=(1.3, 0), borderaxespad=0,
                      edgecolor='black').get_frame().set_linewidth(2)
            a2.legend(legend_list, prop=fp, loc='lower right',
                      bbox_to_anchor=(1.3, 0), borderaxespad=0,
                      edgecolor='black').get_frame().set_linewidth(2)
            # 枠線の太さ
            a1.spines['bottom'].set_linewidth(2)
            a1.spines['top'].set_linewidth(2)
            a1.spines['right'].set_linewidth(2)
            a1.spines['left'].set_linewidth(2)
            a2.spines['bottom'].set_linewidth(2)
            a2.spines['top'].set_linewidth(2)
            a2.spines['right'].set_linewidth(2)
            a2.spines['left'].set_linewidth(2)
            # グラフ保存
            plt.savefig(folder_path + '/' + GRAPH_PATH + '/' + graph_name +
                        '.png')
            print('complete make file: ' + graph_name + '.png')


# グラフの横軸(時間軸：動画の全フレーム数)算出
def make_graph_frame_list(angle_list):
    graph_frame_list = []
    step = 1 / FPS
    for i, angle in enumerate(angle_list):
        temp = np.arange(0, step * len(angle.ca), step)
        if len(temp) > len(angle.ca):
            temp = np.delete(temp, np.s_[len(temp) - 1:])
        graph_frame_list.append(temp)
    return graph_frame_list


# グラフの横軸の目盛り上限算出
def make_graph_frame_qty(frame_list):
    last_frame_list = []
    for frame in frame_list:
        last_frame_list.append(frame[-1])
    graph_frame_qty = max(last_frame_list)
    graph_frame_qty += 10
    return graph_frame_qty


# グラフの線の名前作成
def make_legend_list(data_name_list, person_number_list):
    legend_list = []
    for i, data_name in enumerate(data_name_list):
        if data_name == '何もなし':
            name = '通常歩行'
        elif data_name == '両足':
            name = '両足2.5kg歩行'
        elif data_name == '左足両方':
            name = '左足5kg歩行'
        elif data_name == '左足片方':
            name = '左足2.5kg'
        elif data_name == '右足両方':
            name = '右足5kg歩行'
        elif data_name == '右足片方':
            name = '右足2.5kg歩行'
        legend_list.append(name)
    return legend_list


# グラフ名作成
def make_graph_name(angle_number):
    graph_name = '[' + str(ANGLE_PAIR_LIST[angle_number // 2][0][1])
    graph_name += ','
    graph_name += DataFrameFormat._part_list[
        ANGLE_PAIR_LIST[angle_number // 2][0][1]]
    graph_name += '_angle]&'
    graph_name += '[' + str(ANGLE_PAIR_LIST[angle_number // 2][1][1])
    graph_name += ','
    graph_name += DataFrameFormat._part_list[
        ANGLE_PAIR_LIST[angle_number // 2][1][1]]
    graph_name += '_angle]'
    return graph_name


# サブグラフ名作成
def make_sub_graph_name(angle_number):
    graph_name = str(
        ANGLE_PAIR_LIST[angle_number // 2][angle_number % 2][1])
    graph_name += ','
    graph_name += DataFrameFormat._part_list[
        ANGLE_PAIR_LIST[angle_number // 2][angle_number % 2][1]]
    graph_name += '_angle'
    return graph_name


# 角度の最小値のファイル作成
def make_min_angle_file(angle_list,
                        data_name_list, person_number_list, folder_path,
                        flag):
    for angle_number, data_list in enumerate(angle_list):
        if flag[angle_number // 2]:
            data = make_min_angle_data(data_list,
                                       data_name_list,
                                       person_number_list,
                                       angle_number)
            file_name = make_min_angle_file_name(angle_number)
            with open(folder_path + '/' + MIN_ANGLE_FILE_PATH + '/' +
                      file_name + '.txt', 'wt') as f:
                f.write(data)
            print('complete make file: ' + file_name + '.txt')


# 書き込むデータを作成する
def make_min_angle_data(data_list,
                        data_name_list, person_number_list, angle_number):
    min_angle = calculate_min_angle(data_name_list,
                                    person_number_list,
                                    data_list)
    data = make_sub_graph_name(angle_number) + '\n\n'
    data += min_angle + '\n'
    data = data[:-1]
    return data


# 角度の最小値の差を算出する
def calculate_min_angle(data_name_list, person_number_list, data_list):
    min_angle = ''
    for i, data_i in enumerate(data_list):
        for j, data_j in enumerate(data_list):
            if(i >= j):
                continue
            else:
                min_angle += data_name_list[i]
                min_angle += '(' + str(person_number_list[i]) + ')'
                min_angle += '{' + str(round((data_list[i].min_ca), 2))
                min_angle += '[ °]}-'
                min_angle += data_name_list[j]
                min_angle += '(' + str(person_number_list[j]) + ')'
                min_angle += '{' + str(round((data_list[j].min_ca), 2))
                min_angle += '[ °]}='
                min_angle += '{'
                min_angle += str(round(
                    data_list[i].min_ca - data_list[j].min_ca, 2))
                min_angle += '[ °]}\n'
    return min_angle


# 角度の最小値の差を書き込むファイル名作成
def make_min_angle_file_name(angle_number):
    file_name = make_sub_graph_name(angle_number)
    file_name += '_min_angle'
    return file_name


# 角度の極小値のファイル作成
def make_low_angle_file(angle_list,
                        data_name_list, person_number_list, folder_path,
                        flag):
    for angle_number, data_list in enumerate(angle_list):
        if flag[angle_number // 2]:
            data = make_low_angle_data(data_list,
                                       data_name_list,
                                       person_number_list,
                                       angle_number)
            file_name = make_low_angle_file_name(angle_number)
            with open(folder_path + '/' + LOW_ANGLE_FILE_PATH + '/' +
                      file_name + '.txt', 'wt') as f:
                f.write(data)
            print('complete make file: ' + file_name + '.txt')


# 書き込むデータを作成する
def make_low_angle_data(data_list,
                        data_name_list, person_number_list, angle_number):
    low_angle = calculate_low_angle(data_name_list,
                                    person_number_list,
                                    data_list)
    data = make_sub_graph_name(angle_number) + '\n\n'
    data += low_angle + '\n'
    data = data[:-1]
    return data


# 角度の極小値の差を算出する
def calculate_low_angle(data_name_list, person_number_list, data_list):
    low_angle = ''
    for i, data_i in enumerate(data_list):
        for j, data_j in enumerate(data_list):
            if(i >= j):
                continue
            else:
                low_angle += data_name_list[i]
                low_angle += '(' + str(person_number_list[i]) + ')'
                low_angle += '{' + str(round((data_list[i].low_ca), 2))
                low_angle += '[ °]}-'
                low_angle += data_name_list[j]
                low_angle += '(' + str(person_number_list[j]) + ')'
                low_angle += '{' + str(round((data_list[j].low_ca), 2))
                low_angle += '[ °]}='
                low_angle += '{'
                low_angle += str(round(
                    data_list[i].low_ca - data_list[j].low_ca, 2))
                low_angle += '[ °]}\n'
    return low_angle


# 角度の極小値の差を書き込むファイル名作成
def make_low_angle_file_name(angle_number):
    file_name = make_sub_graph_name(angle_number)
    file_name += '_low_angle'
    return file_name


# 周期のファイル作成
def make_T_file(angle_list, data_name_list, person_number_list, folder_path,
                flag):
    for angle_number, data_list in enumerate(angle_list):
        if flag[angle_number // 2]:
            data = make_T_data(data_list, data_name_list, person_number_list,
                               angle_number)
            file_name = make_T_file_name(angle_number)
            with open(folder_path + '/' + T_FILE_PATH + '/' +
                      file_name + '.txt', 'wt') as f:
                f.write(data)
            print('complete make file: ' + file_name + '.txt')


# 書き込むデータを作成する
def make_T_data(data_list, data_name_list, person_number_list, angle_number):
    T = calculate_T(data_name_list, person_number_list, data_list)
    data = make_sub_graph_name(angle_number) + '\n\n'
    data += T + '\n'
    data = data[:-1]
    return data


# 周期の差を算出する
def calculate_T(data_name_list, person_number_list, data_list):
    T = ''
    for i, data_i in enumerate(data_list):
        for j, data_j in enumerate(data_list):
            if(i >= j):
                continue
            else:
                T += data_name_list[i]
                T += '(' + str(person_number_list[i]) + ')'
                T += '{' + str(round((data_list[i].T), 2)) + '[s]}-'
                T += data_name_list[j]
                T += '(' + str(person_number_list[j]) + ')'
                T += '{' + str(round((data_list[j].T), 2)) + '[s]}='
                T += '{'
                T += str(round(
                    data_list[i].T - data_list[j].T, 2))
                T += '[s]}\n'
    return T


# 周期時間の差を書き込むファイル名作成
def make_T_file_name(angle_number):
    file_name = make_sub_graph_name(angle_number)
    file_name += '_T'
    return file_name


# main
def main():
    data_list_list = []
    flag_list = []
    angle_data_list_list = []
    # グラフ化するファイルの情報を読み込む
    data_name_list, person_number_list = input_file_data()
    for angle_list in ANGLE_PAIR_LIST:
        for angle in angle_list:
            for part_number in angle:
                # 読み込んだ情報を元にcsvファイルを探す　見つけたファイル名を取得
                csv_file_name_list = serch_csv(data_name_list,
                                               person_number_list,
                                               part_number)
                # 見つけたファイルを読み込む
                data_list_list.append(read_csv(csv_file_name_list))
    data_list = organize_by_data(data_list_list)
    for i, data_name in enumerate(data_name_list):
        flag_list.append(deletion_judgment1(data_list[i],
                                            data_name, person_number_list[i]))
        angle_data_list_list.append(calculate_angle(data_list[i]))
        start_and_stop_list = serch_start_and_stop(angle_data_list_list[i])
        for j, start_and_stop in enumerate(start_and_stop_list):
            angle_data_list_list[i][j].cut_off_start_end(start_and_stop[0],
                                                         start_and_stop[1])
    pair_angle_flag_list = organize_by_data(flag_list)
    flag = deletion_judgment2(pair_angle_flag_list)
    angle_list = organize_by_data(angle_data_list_list)
    # ファイル保存フォルダ作成
    jtoc.make_folder(RESULT_FOLDER_PATH)
    folder_path = make_folder_path(data_name_list, person_number_list)
    jtoc.make_folder(folder_path)
    jtoc.make_folder(folder_path + '/' + GRAPH_PATH)
    make_graph(angle_list,
               data_name_list, person_number_list, folder_path,
               flag)
    if len(data_name_list) >= 2:
        jtoc.make_folder(folder_path + '/' + MIN_ANGLE_FILE_PATH)
        make_min_angle_file(angle_list,
                            data_name_list, person_number_list, folder_path,
                            flag)
        jtoc.make_folder(folder_path + '/' + LOW_ANGLE_FILE_PATH)
        make_low_angle_file(angle_list,
                            data_name_list, person_number_list, folder_path,
                            flag)
        jtoc.make_folder(folder_path + '/' + T_FILE_PATH)
        make_T_file(angle_list,
                    data_name_list, person_number_list, folder_path,
                    flag)


if __name__ == '__main__':
    main()