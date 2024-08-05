# Generative-Articulated-Object-s-

https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris

```
sudo apt-get install libgl1-mesa-dri
```

```
conda env create -f env.yaml
```

transformer 的数据集存储在 redis 中，所以需要安装 redis @see https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/

# 下载数据集

https://sapien.ucsd.edu/downloads
解压进 `dataset/raw`

```
(gao) (base) shuyumo@SYM:~/research/GAO/dataset/raw$ pwd
/home/shuyumo/research/GAO/dataset/raw
(gao) (base) shuyumo@SYM:~/research/GAO/dataset/raw$ ls
100013  100340  100712  100905  101114  101450  101842  102202  102567 ..............
```

# commands

- 准备数据集

  ```
  cd process_data/
  python 1_extract_from_raw_dataset.py
  python 2_generate_onet_dataset.py
  ```

- 训练 ONet

  ```
  python train-onet.py -c configs/onet/train-onet_v2.yaml
  ```

- 利用训练好的 ONet 生成 transformer 数据集

  ```
  python 4.1_generate_screenshot.py
  python 4.2_get_text_description.py
  python 4.3_generate_text_latent.py
  python 4.4_generate_transformer_dataset.py
  ```

- 训练 transformer

  ```
  python main-transformer.py -c configs/transformer/train-default-v3.yaml
  ```

