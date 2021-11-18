def split_point(pt_str):
  res = pt_str.split(',')
  return [float(x) for x in res]

def read_points(pc_path):
  tmp_res = []
  with open(pc_path, 'r') as file:
    tmp_res = file.readlines()

  return [tuple(split_point(x)) for x in tmp_res]
