# SZO
the implementation of [Sharpness-aware Zeroth-order Optimization for Graph Transformers] (IJCAI-25)


## implementation


model: ./model/graphtransformer.py

optimizer: ./optimizer/szo.py

more implementation of graph transformers can see [GraphSAM](https://github.com/YL-wang/GraphSAM/tree/graphsam)


```python
from optimizer import SZO

model = YourModel()
criterion = YourCriterion()
base_optimizer = YourBaseOptimizer

optimizer = SZO(
    params=model.parameters(),
    lr=0.01
    estimator="rge"
)

for epoch in range(epochs):
  i=0
  for batch in (train_loader):
    def closure():
	output = model(input)
	loss = loss_f(output, target)
	loss.backward()
	return loss

    if i==0:
	output = model(input)
	loss=loss_f(output,target)
	optimizer.step(i, epoch, closure, loss)
    else:
	optimizer.step(i, epoch, closure)
    loss = optimizer.get_loss()
    optimizer.zero_grad()
```