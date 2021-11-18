class Node(object):
  def __init__(self, point):
    assert len(point) == 3

    self.x = point[0]
    self.y = point[1]
    self.z = point[2]
    self.left = None
    self.right = None

  def get_position(self):
    return (self.x, self.y, self.z)

  def get_children(self):
    return (self.left, self.right)