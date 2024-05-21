# Generative-Articulated-Object-s-

https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris
```
sudo apt-get install libgl1-mesa-dri
```

## 准备数据集
```
1_extract_from_raw_dataset.py
2_convert_to_onet_dataset.py
3_evaluate_latent_code.py
4_push_to_redis.py
```


## 训练
```
python main.py -c configs/default.yaml
```

## 看点好的
./logs/interp_test/interp.gif
![./logs/interp_test/interp.gif](./logs/interp_test/interp.gif)

./logs/interp_test/usb-body-noise-higher.gif
![./logs/interp_test/usb-body-noise-higher.gif](./logs/interp_test/usb-body-noise-higher.gif)

./logs/interp_test/usb-cap-noise-higher.gif
![./logs/interp_test/usb-cap-noise-higher.gif](./logs/interp_test/usb-cap-noise-higher.gif)