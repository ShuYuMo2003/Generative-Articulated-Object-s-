# Generative-Articulated-Object-s-

https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris
```
sudo apt-get install libgl1-mesa-dri
```

```
conda env create -f env.yaml
```

transformer 的数据集存储在 redis 中，所以需要安装 redis @see https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/

# commands
 - 准备数据集
    ```
    python main-onet.py -c configs/onet/prepare_transformer_dataset.yaml
    ```

 - 训练 ONet
    ```
    python train-onet.py -c configs/onet/train-onet.yaml
    ```

 - 利用训练好的 ONet 生成 transformer 数据集
    ```
    python main-onet.py -c configs/onet/prepare_transformer_dataset.yaml
    ```

 - 训练 transformer
    ```
    python main-transformer.py -c configs/transformer/train-default-v2.yaml
    ```

 - 推理 transformer
    ```
    python main-transformer.py -c configs/transformer/eval-default.yaml
    ```
