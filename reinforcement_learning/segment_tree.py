import unittest
import random

class SumSegmentTree:

    def __init__(self, size: int) -> None:
        '''
        Root is on index 1.
        '''
        # Construct tree
        # Find the smallest power of 2 >= size
        tree_size = 1
        while tree_size < size:
            tree_size = tree_size * 2
        self.size = tree_size
        self.tree = [0 for _ in range(2 * self.size)]

    def _recur_sum(self, start_idx, end_idx, node_start, node_end, node):
        if start_idx == node_start and end_idx == node_end:
            return self.tree[node]

        node_mid = (node_start + node_end) // 2
        if node_mid >= end_idx:
            # Move to the left child node
            return self._recur_sum(start_idx, end_idx, node_start, node_mid, node * 2)
        else:
            if node_mid < start_idx:
                # Move to the right child node
                return self._recur_sum(start_idx, end_idx, node_mid + 1, node_end, node * 2 + 1)
            else:
                return self._recur_sum(
                    start_idx,
                    node_mid,
                    node_start,
                    node_mid,
                    node * 2
                ) + self._recur_sum(
                    node_mid+1,
                    end_idx,
                    node_mid+1,
                    node_end,
                    node * 2 + 1
                )

    def sum(self, start: int = 0, end = None) -> float:
        '''
        returns sum in range [start, end], INCLUSIVE
        '''
        if end is None:
            end = self.size - 1
        return self._recur_sum(start, end, 0, self.size - 1, 1)


    def retrieve(self, value):
        '''
        Return the smallest index i, such that sum(0, i) >= value
        '''
        tree_ptr = 1
        assert value <= self.tree[1], str(value) + ' ' + str(self.tree[1])
        node_start = 0
        node_end = self.size - 1
        while node_start != node_end:
            node_mid = (node_start + node_end) // 2
            node_value = self.tree[tree_ptr]
            l_value = self.tree[tree_ptr * 2]
            r_value = self.tree[tree_ptr * 2 + 1]
            if value > l_value:
                value = value - l_value
                node_start = node_mid + 1
                tree_ptr = tree_ptr * 2 + 1
            elif value == l_value:
                return node_mid
            else:
                node_end = node_mid
                tree_ptr = tree_ptr * 2
        return node_start



    def __setitem__(self, idx, val):
        tree_idx = idx + self.size
        tree_ptr = tree_idx
        self.tree[tree_ptr] = val
        tree_ptr = tree_ptr // 2
        while tree_ptr >= 1:
            self.tree[tree_ptr] = self.tree[2 * tree_ptr] + \
                self.tree[2 * tree_ptr + 1]
            tree_ptr = tree_ptr // 2


    def __getitem__(self, idx):
        return self.tree[idx + self.size]

class TestSumSegmentTree(unittest.TestCase):

    def _test_correctness(self, length, trials=3):
        array = [0 for _ in range(length)]
        tree = SumSegmentTree(length)
        for i in range(length):
            array[i] = random.randint(0, 1000)
            tree[i] = array[i]
        for trial in range(trials):
            start = random.randint(0, length - 1)
            end = random.randint(start, length - 1)
            self.assertTrue(tree.sum(start, end) == sum(array[start : end+1]))

    def test_correctness(self):
        self._test_correctness(7, 10)
        self._test_correctness(19, 20)
        self._test_correctness(12, 129)

    def test_retrieve(self):
        array = [0 for _ in range(7)]
        tree = SumSegmentTree(7)
        for i in range(7):
            array[i] = i
            tree[i] = array[i]
        self.assertTrue(tree.retrieve(0) == 0)
        self.assertTrue(tree.retrieve(1) == 1)
        self.assertTrue(tree.retrieve(2) == 2)
        self.assertTrue(tree.retrieve(3) == 2)
        self.assertTrue(tree.retrieve(4) == 3)
        self.assertTrue(tree.retrieve(8) == 4)
        self.assertTrue(tree.retrieve(12) == 5)
        self.assertTrue(tree.retrieve(20) == 6)

        array = [0 for _ in range(5)]
        tree = SumSegmentTree(5)
        for i in range(5):
            array[i] = i
            tree[i] = array[i]
        self.assertTrue(tree.retrieve(0) == 0)
        self.assertTrue(tree.retrieve(1) == 1)
        self.assertTrue(tree.retrieve(2) == 2)
        self.assertTrue(tree.retrieve(3) == 2)
        self.assertTrue(tree.retrieve(5) == 3)
        self.assertTrue(tree.retrieve(9) == 4)


if __name__ == '__main__':
    unittest.main()
