# CyberHarem

Cyber Harem of All the Waifus in Games, Mua~

## Train A PLora

Use amiya's online dataset from [CyberHarem](https://huggingface.co/CyberHarem).

```python
from ditk import logging

from cyberharem.train import train_plora

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    train_plora(
        'amiya',
    )

```

The experiment directory will be at `runs/amiya_arknights`

If you need to use your own local dataset, just like this

```python
from ditk import logging

from cyberharem.train import train_plora

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    train_plora(
        source='/path/to/dataset',
        name='trigger_word',
        steps=1000,
        workdir='runs/trigger_word',
    )

```

## Publish the Trained Model

```shell
python -m cyberharem.publish huggingface -w runs/amiya_arknights
```

The model and preview images will be deployed to huggingface repo `CyberHarem/amiya_arknights`.


