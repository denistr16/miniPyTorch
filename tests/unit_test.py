import unittest
import numpy as np

from mini_torch.tensor import Tensor as T
from tests.unit_test_data import test_data


class MathTest(unittest.TestCase):
    def test_add(self):
        x1 = T.from_numpy('x1', test_data['x1'], required_grad=False)
        x2 = T.from_numpy('x2', test_data['x2'], required_grad=False)

        w0 = T.from_numpy('w0', test_data['w0'])
        w1 = T.from_numpy('w1', test_data['w1'])
        w2 = T.from_numpy('w2', test_data['w2'])

        f = w0 + w1 ** 2 * x1 + w2**2 + x2
        f_t = test_data['w0'] + test_data['w1']**2 * test_data['x1'] + \
              test_data['w2']**2 + test_data['x2']

        self.assertEqual(f.data, f_t)


if __name__ == '__main__':
    unittest.main()
