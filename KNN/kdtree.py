import argparse
import copy
import utils
import numpy as np
from node import Node
from bisect import bisect
from time import time

class KDTree(object):
  """Construct a kd tree"""
  def __init__(self, file_path):
    self.pc_path = file_path
    self.dim = 3
    points = utils.read_points(self.pc_path)
    points = np.array(points, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    self.points = np.unique(points)
    index_x = np.argsort(self.points, order=('x','y','z'))
    index_y = np.argsort(self.points, order=('y','z','x'))
    index_z = np.argsort(self.points, order=('z','x','y'))
    self.root = self.build_tree(self.points, index_x, index_y, index_z, 0)

  def partition(self, points, index, median_point_index, axis):
    median_point = points[median_point_index]
    index_lower = []
    index_upper = []
    for i in index:
      if i == median_point_index:
        continue
      if points[i][axis] < median_point[axis]:
        index_lower.append(i)
      else:
        index_upper.append(i)
    return index_lower, index_upper

  def build_tree(self, points, index_x, index_y, index_z, level):
    # print (index_x, index_y, index_z)
    assert len(index_x) == len(index_y) == len(index_z)
    length = len(index_x)

    if length == 0:
      return None
    elif length == 1:
      return Node(points[index_x[0]])
    else:
      if level == 0: 
        median_point_index = index_x[length/2]
        index_x_lower = copy.deepcopy(index_x[:length/2])
        index_x_upper = copy.deepcopy(index_x[length/2+1:])
        index_y_lower, index_y_upper = self.partition(points, index_y, median_point_index, level)
        index_z_lower, index_z_upper = self.partition(points, index_z, median_point_index, level)

      elif level == 1:
        median_point_index = index_y[length/2]
        index_y_lower = copy.deepcopy(index_y[:length/2])
        index_y_upper = copy.deepcopy(index_y[length/2+1:])
        index_z_lower, index_z_upper = self.partition(points, index_z, median_point_index, level)
        index_x_lower, index_x_upper = self.partition(points, index_x, median_point_index, level)

      elif level == 2:
        median_point_index = index_z[length/2]
        index_z_lower = copy.deepcopy(index_z[:length/2])
        index_z_upper = copy.deepcopy(index_z[length/2+1:])
        index_x_lower, index_x_upper = self.partition(points, index_x, median_point_index, level)
        index_y_lower, index_y_upper = self.partition(points, index_y, median_point_index, level)

      del index_x, index_y, index_z 
      res = Node(points[median_point_index])
      res.left = self.build_tree(points, index_x_lower, index_y_lower, index_z_lower, (level+1)%self.dim)
      res.right = self.build_tree(points, index_x_upper, index_y_upper, index_z_upper, (level+1)%self.dim)
      return res

  def nearest_neighbor(self, query, curr_node, level):
    if curr_node.left is None and curr_node.right is None:
      sqr_dis = np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)
      return [curr_node], sqr_dis

    curr_best_nodes = [curr_node] 
    curr_best_sqr_dis = np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)

    if curr_node.left is not None:
      best_left_children, best_left_sqr_dis = self.nearest_neighbor(query, curr_node.left, (level+1)%self.dim)
      if best_left_sqr_dis < curr_best_sqr_dis:
        curr_best_nodes = best_left_children
        curr_best_sqr_dis = best_left_sqr_dis
      elif best_left_sqr_dis == curr_best_sqr_dis:
        curr_best_nodes += best_left_children

    if curr_node.right is not None:
      axis_sqr_dis = (query[level] - curr_node.get_position()[level]) ** 2
      if axis_sqr_dis < curr_best_sqr_dis:
        best_right_children, best_right_sqr_dis = self.nearest_neighbor(query, curr_node.right, (level+1)%self.dim)
        if best_right_sqr_dis < curr_best_sqr_dis:
          curr_best_nodes = best_right_children
          curr_best_sqr_dis = best_right_sqr_dis
        elif best_right_sqr_dis == curr_best_sqr_dis:
          curr_best_nodes += best_right_children

    return curr_best_nodes, curr_best_sqr_dis

  def k_nearest_neighbors(self, query, curr_node, k, level):

    if curr_node.left is None and curr_node.right is None:
      sqr_dis = np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)
      return [curr_node], [sqr_dis]
    curr_best_nodes = [curr_node]
    curr_best_sqr_dis = [np.sum((np.array(query) - np.array(curr_node.get_position())) ** 2)]

    if curr_node.left is not None:
      best_left_children, best_left_sqr_dis = self.k_nearest_neighbors(query, curr_node.left, k, (level+1)%self.dim)

      if len(best_left_children) < k or curr_best_sqr_dis[0] < best_left_sqr_dis[-1]:
        idx = bisect(best_left_sqr_dis, curr_best_sqr_dis[0])
        best_left_sqr_dis.insert(idx, curr_best_sqr_dis[0])
        best_left_children.insert(idx, curr_best_nodes[0])

      curr_best_nodes = best_left_children[:k]
      curr_best_sqr_dis = best_left_sqr_dis[:k]

    if curr_node.right is not None:
      axis_sqr_dis = (query[level] - curr_node.get_position()[level]) ** 2
      if len(curr_best_nodes) < k or curr_best_sqr_dis[-1] > axis_sqr_dis:
        best_right_children, best_right_sqr_dis = self.k_nearest_neighbors(query, curr_node.right, k, (level+1)%self.dim)
        tmp_dis = []
        tmp_nodes = []
        curr_idx = 0
        right_idx = 0
        while right_idx<len(best_right_sqr_dis) and curr_idx<len(curr_best_sqr_dis) and len(tmp_dis)<k:
          if best_right_sqr_dis[right_idx] < curr_best_sqr_dis[curr_idx]:
            tmp_dis.append(best_right_sqr_dis[right_idx])
            tmp_nodes.append(best_right_children[right_idx])
            right_idx += 1
          else:
            tmp_dis.append(curr_best_sqr_dis[curr_idx])
            tmp_nodes.append(curr_best_nodes[curr_idx])
            curr_idx += 1

        tmp_size = len(tmp_dis)
        if tmp_size < k:
          if curr_idx == len(curr_best_sqr_dis):
            tmp_dis += best_right_sqr_dis[right_idx:min(right_idx+k-tmp_size,len(best_right_sqr_dis))]
            tmp_nodes += best_right_children[right_idx:min(right_idx+k-tmp_size,len(best_right_children))]
          elif right_idx == len(best_right_sqr_dis):
            tmp_dis += curr_best_sqr_dis[curr_idx:min(curr_idx+k-tmp_size,len(curr_best_sqr_dis))]
            tmp_nodes += curr_best_nodes[curr_idx:min(curr_idx+k-tmp_size,len(curr_best_nodes))]

        curr_best_nodes = tmp_nodes
        curr_best_sqr_dis = tmp_dis

    return curr_best_nodes, curr_best_sqr_dis



def main(pc_path):
  tree = KDTree(pc_path)
  k = 3
  start = time()
  print ('The %d nearest neighbors for query point (8.0, 3.0, 1.0):' % k)
  nn, _ = tree.k_nearest_neighbors((8.0, 3.0, 1.0), tree.root, k, 0)
  for node in nn:
    print (node.get_position())
  finish = time()
  print("Busqueda:",(finish-start)*1000)


# Para correr el programa: python kdtree.py -i ./datos/test100.csv
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Construct a KD-tree from a list of 3D points")
  parser.add_argument('-i', '--input', metavar='', type=str, help="The path to input point cloud file", required=True)
  args = parser.parse_args()
  main(args.input)
