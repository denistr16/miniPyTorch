# miniPyTorch
This is a toy torch-like framework for educational purposes.    
The central idea of this project is to implement autogradient mechanics and show how it works. 

![](https://github.com/denistr16/miniPyTorch/blob/master/media/chain-rule.jpg)


## Installation

OS X or Linux-like:

```sh
chmod +x install.sh
./install.sh
```

## Usage example

To avoid dependency conflict we will keep dev under python virtual environment:
```sh
source venv/bin/activate
```

Let's take a look at how the simplest function can be implemented within our framework 
and compute it's gradient:
```python
import numpy as np
from mini_torch.tensor import Tensor as T

w1 = T.from_numpy('w1', np.array([-0.91]).astype(float))
w0 = T.from_numpy('w0', np.array([1.5]).astype(float))

x = T.from_numpy('x', np.array([2.]).astype(float), required_grad=False)

y = w0 + w1*x

print(y)

y.backward()

print('grad w0', w0.grad)
print('grad w1', w1.grad)
```


## Meta

Denistr16 – [@github](https://github.com/denistr16)

Distributed under the MIT license. 
See [``LICENSE``](https://github.com/denistr16/miniPyTorch/blob/master/LICENSE.md) for more information.

[https://github.com/denistr16/miniPyTorch](https://github.com/denistr16/miniPyTorch)

## Contributing

1. Fork it (<https://github.com/denistr16/miniPyTorch/fork>)
2. Create your feature branch (`git checkout -b feature/myNewFeature`)
3. Commit your changes (`git commit -am 'Add some myNewFeature'`)
4. Push to the branch (`git push origin feature/myNewFeature`)
5. Create a new Pull Request
