#!/usr/bin/env bash


for i in `ls *_crop.npy`; do
    mv $i ${i%_crop.npy}.npy
done