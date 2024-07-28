#!/bin/bash
kill -9 $(ps -aux | grep "configs/transformer/train-default-v3.yaml" | cut -c12-16)