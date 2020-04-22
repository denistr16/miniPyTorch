import numpy as np


class Tensor:
    def __init__(self, name, shape=None, data=None, required_grad=True, grad_value=None, prev=None):

        self.data = None
        if data is not None:
            self.data = data

        if (shape is not None) and (data is None):
            self.data = np.zeros(shape)

        self.name = name
        self.prev = prev

        self.grad = None
        self.back_prop_grad = {}

        if required_grad:
            if grad_value is None:
                self.grad = {self.name: 1.}
            else:
                self.grad = grad_value

    @classmethod
    def from_numpy(cls, name: str, data: np.array, required_grad=True) -> 'Tensor':
        assert type(data) == np.ndarray
        return cls(name=name, shape=None, data=data, required_grad=required_grad)

    @classmethod
    def from_tensor(cls, name: str, tensor, chained=False) -> 'Tensor':
        required_grad = tensor.grad is not None
        if not chained:
            return cls(name=name, shape=None, data=tensor.data, required_grad=required_grad,
                       grad_value=tensor.grad, prev=tensor.prev)
        else:
            grad_value = {tensor.name: 1.}
            return cls(name=name, shape=None, data=tensor.data, required_grad=required_grad,
                       grad_value=grad_value, prev={tensor.name: tensor})

    def __add__(self, obj):
        grad_value = {}

        if self.grad is not None:
            grad_value[self.name] = 1

        if obj.grad is not None:
            grad_value[obj.name] = 1

        prev = {self.name: self, obj.name: obj}
        name = 'add'
        data = np.add(self.data, obj.data)
        required_grad = len(grad_value) != 0

        return Tensor(name=name, shape=None, data=data,
                      required_grad=required_grad, grad_value=grad_value, prev=prev)

    def __sub__(self, obj):
        # minus_one = Tensor.from_numpy(name='minus', data=-1 * np.ones(obj.shape()), required_grad=False)
        # return Tensor.from_tensor(name='sub', tensor=self + obj * minus_one, chained=True)
        grad_value = {}

        if self.grad is not None:
            grad_value[self.name] = 1

        if obj.grad is not None:
            grad_value[obj.name] = -1

        prev = {self.name: self, obj.name: obj}
        name = 'sub'
        data = np.subtract(self.data, obj.data)
        required_grad = len(grad_value) != 0

        return Tensor(name=name, shape=None, data=data,
                      required_grad=required_grad, grad_value=grad_value, prev=prev)

    def __mul__(self, obj):
        grad_value = {}

        if self.grad is not None:
            grad_value[self.name] = obj.data

        if obj.grad is not None:
            grad_value[obj.name] = self.data

        prev = {self.name: self, obj.name: obj}
        name = 'mul'
        data = np.multiply(self.data, obj.data)
        required_grad = len(grad_value) != 0

        return Tensor(name=name, shape=None, data=data,
                      required_grad=required_grad, grad_value=grad_value, prev=prev)

    def __truediv__(self, obj):
        return Tensor.from_tensor(name='div', tensor=self*(obj**-1), chained=True)

    def __pow__(self, value):
        grad_value = {}

        if self.grad is not None:
            grad_value[self.name] = value * np.power(self.data, value-1)

        prev = {self.name: self}
        name = 'pow'
        data = np.power(self.data, value)
        required_grad = len(grad_value) != 0

        return Tensor(name=name, shape=None, data=data,
                      required_grad=required_grad, grad_value=grad_value, prev=prev)

    def shape(self):
        return self.data.shape

    def __str__(self):
        rep = "{}\n".format(str(type(self)))
        rep += "{}\n".format(str(self.data))
        rep += "shape: {}\n".format(str(self.shape()))
        return rep

    def backward_a(self):
        if self.prev is None:
            return self.grad[self.name]
        else:
            result = []
            for k, v in self.prev.items():
                if v.grad is not None:
                    result.append(self.grad[k] * self.prev[k].backward())
                else:
                    pass
            return result

    def backward_b(self, d=dict()):
        if self.prev is None:
            d[self.name] = self.grad[self.name]
            return d
        else:

            for name, obj in self.prev.items():
                if obj.grad is not None:
                    d = self.prev[name].backward_b(d)
                    if name in d.keys():
                        d[name] = d[name] * self.grad[name]

                else:
                    pass
            return d

    # TODO write zero_grad()
    def backward(self):
        if self.prev is None:
            self.back_prop_grad[self.name] = self.grad[self.name]
            return self.back_prop_grad
        else:
            for prev_name, prev_obj in self.prev.items():
                if prev_obj.grad is not None:
                    back_prop_grad = prev_obj.backward()
                    for k, v in back_prop_grad.items():
                        back_prop_grad[k] = v * self.grad[prev_name]
                    self.back_prop_grad.update(back_prop_grad)
                else:
                    pass
            return self.back_prop_grad
