import unittest
import torch

from mini_torch.tensor import Tensor as T
from tests.unit_test_data import test_data

x1 = T.from_numpy('x1', test_data['x1'], required_grad=False)
x2 = T.from_numpy('x2', test_data['x2'], required_grad=False)

w0 = T.from_numpy('w0', test_data['w0'])
w1 = T.from_numpy('w1', test_data['w1'])
w2 = T.from_numpy('w2', test_data['w2'])


class MathTest(unittest.TestCase):
    def test_add(self):
        f = w0 + x1
        f_t = test_data['w0'] + test_data['x1']
        self.assertEqual(f.data, f_t)

    def test_sub(self):
        f = x1 - w2
        f_t = test_data['x1'] - test_data['w2']
        self.assertEqual(f.data, f_t)

    def test_mul(self):
        f = x1 * w1
        f_t = test_data['x1'] * test_data['w1']
        self.assertEqual(f.data, f_t)

    def test_div(self):
        f = x1 / w1
        f_t = test_data['x1'] / test_data['w1']
        self.assertAlmostEqual(f.data[0], f_t[0])

    def test_pow(self):
        f = w1 ** -3
        f_t = test_data['w1'] ** -3
        self.assertEqual(f.data, f_t)

    def test_poly(self):
        f = w0 + w1 ** 2 * x1 - w2 ** 2 * x2
        f_t = test_data['w0'] + test_data['w1'] ** 2 * test_data['x1'] - test_data['w2'] ** 2 * test_data['x2']

        self.assertEqual(f.data, f_t)


class GradTest(unittest.TestCase):
    def test_grad(self):
        f = w0 + w1 ** 2 * x1 - w2 ** 2 * x2
        f.zero_grad()
        f.backward()

        t_w0 = torch.tensor(test_data['w0'], requires_grad=True)
        t_w1 = torch.tensor(test_data['w1'], requires_grad=True)
        t_w2 = torch.tensor(test_data['w2'], requires_grad=True)
        t_x1 = torch.tensor(test_data['x1'])
        t_x2 = torch.tensor(test_data['x2'])

        t_y = t_w0 + t_w1 ** 2 * t_x1 - t_w2 ** 2 * t_x2
        t_y.backward()

        self.assertEqual(w0.grad['w0'], t_w0.grad.data.numpy())
        self.assertEqual(w1.grad['w1'], t_w1.grad.data.numpy())
        self.assertEqual(w2.grad['w2'], t_w2.grad.data.numpy())


if __name__ == '__main__':
    unittest.main()
