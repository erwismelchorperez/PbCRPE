class PBCRENode:
    def __init__(self, node_id, depth):
        self.node_id = node_id
        self.depth = depth

        # estructura
        self.is_leaf = True
        self.split_feature = None
        self.threshold = None
        self.left = None
        self.right = None

        # estadísticas
        self.samples = 0
        self.class_counts = None
        self.predicted_class = None

        # explicabilidad futura
        self.extra_info = None
        self.leaf_threshold = 0.5   # 👈 NUEVO

    def set_split(self, feature, threshold, left, right, extra_info=None):
        self.is_leaf = False
        self.split_feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.extra_info = extra_info

    def decide(self, row):
        if self.is_leaf:
            return None
        return self.left if row[self.split_feature] <= self.threshold else self.right
