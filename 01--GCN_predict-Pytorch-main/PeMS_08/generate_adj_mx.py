import numpy as np
import csv
import pickle

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA

direction = True
distance_df_filename, num_of_vertices, id_filename = "datasets/raw_data/PeMS_08/PeMS_08.csv", 170, None
if direction:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None)
else:
    adj_mx, distance_mx = get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None)
# TODO: the self loop is missing
add_self_loop = False
if add_self_loop:
    adj_mx = adj_mx + np.identity(adj_mx.shape[0])
    distance_mx = distance_mx + np.identity(distance_mx.shape[0])
pickle.dump(adj_mx, open("datasets/sensor_graph/adj_mx_08.pkl", 'wb'))
pickle.dump(distance_mx, open("datasets/sensor_graph/adj_mx_08_distance.pkl", 'wb'))
