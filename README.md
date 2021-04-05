# CF-Net : Deep Coupled Feedback Network for Joint Exposure Fusion and Super-Resolution
- This is the official repository of the paper "Deep Coupled Feedback Network for Joint Exposure Fusion and Image Super-Resolution" from **IEEE Transactions on Image Processing 2021**. [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/9357931, "Paper Link")[[PDF Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9357931)
- We have conducted a live streaming on Extreme Mart Platform, the Powerpoint file can be downloaded from [[PPT Link]](https://kdocs.cn/l/coxDwl57PbVi
).

![framework](./imgs/framework.png)

## 1. Environment
- Python >= 3.5
- PyTorch >= 0.4.1 is recommended
- opencv-python
- tqdm
- Matlab

## 2. Dataset
The training data and testing data is from the [[SICE dataset]](https://github.com/csjcai/SICE, "Official SICE"). Or you can download the datasets from our [[Google Drive Link]](https://drive.google.com/drive/folders/1Ik0D2pf93aLOlexevpAE5ftckMTQscZo?usp=sharing., "Ours").

## 3. Test
1. Clone this repository:
    ```
    git clone https://github.com/ytZhang99/CF-Net.git
    ```
2. Place the low-resolution over-exposed images and under-exposed images in `dataset/test_data/lr_over` and `dataset/test_data/lr_under`, respectively.
3. Run the following command for 2 or 4 times SR and exposure fusion:
    ```
    python main.py --test_only --scale 2 --model model_x2.pth
    python main.py --test_only --scale 4 --model model_x4.pth
    ```
4. Finally, you can find the Super-resolved and Fused results in `./test_results`.

## 4. Training
For some reason, we haven't released the training code.

If you want to get access to the training code, you can email `yutongzhang@buaa.edu.cn` for the training methods and materials. 

## 5. Citation
If you find our work useful in your research or publication, please cite our work:
```
@article{deng2021deep,
  title={Deep Coupled Feedback Network for Joint Exposure Fusion and Image Super-Resolution.},
  author={Deng, Xin and Zhang, Yutong and Xu, Mai and Gu, Shuhang and Duan, Yiping},
  journal={IEEE Transactions on Image Processing: a Publication of the IEEE Signal Processing Society},
  year={2021}
}
```
