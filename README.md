# NEWS
* Apr 1, 2020 -> NEW paper on Space-Time Super-Resolution [STARnet](https://github.com/alterzero/STARnet) (to appear in CVPR2020)
* Jan 10, 2019 -> Added model used for PIRM2018, and support Pytorch >= 1.0.0
* Mar 25, 2019 -> Paper on Video Super-Resolution [RBPN](https://github.com/alterzero/RBPN-PyTorch) (CVPR2019)
* Apr 12, 2019 -> Added [Extension of DBPN](https://arxiv.org/abs/1904.05677) paper and model. 

# Deep Back-Projection Networks for Super-Resolution (CVPR2018)

## Winner (1st) of [NTIRE2018](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf) Competition (Track: x8 Bicubic Downsampling)

## Winner of [PIRM2018](https://arxiv.org/pdf/1809.07517.pdf) (1st on Region 2, 3rd on Region 1, and 5th on Region 3)

Project page: https://alterzero.github.io/projects/DBPN.html

We also provide original [Caffe implementation](https://github.com/alterzero/DBPN-caffe)

## Pretrained models and Results
Pretrained models (DBPNLL) and results can be downloaded from this link! 
https://drive.google.com/drive/folders/1ahbeoEHkjxoo4NV1wReOmpoRWbl448z-?usp=sharing

## Dependencies
* Python 3.5
* PyTorch >= 1.0.0

## Model types
1. "DBPN" -> use T = 7
2. "DBPNLL" -> use T = 10
3. PIRM Model -> "DBPNLL" with adversarial loss
4. "DBPN-RES-MR64-3" -> [improvement of DBPN](https://arxiv.org/abs/1904.05677) with recurrent process + residual learning

##########HOW TO##########

#Training

    ```python3
    main.py
    ```

#Testing

    ```python3
    eval.py
    ```

#Training GAN for PIRM2018

    ```python3
    main_gan.py
    ```

#Testing GAN for PIRM2018

    ```python3
    eval_gan.py
    ```

![DBPN](http://www.toyota-ti.ac.jp/Lab/Denshi/iim/members/muhammad.haris/projects/DBPN.png)


## Citations
If you find this work useful, please consider citing it.
```
@inproceedings{DBPN2018,
  title={Deep Back-Projection Networks for Super-Resolution},
  author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}

@article{DBPN2019,
  title={Deep Back-Projection Networks for Single Imaage Super-Resolution},
  author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
  journal={arXiv preprint arXiv:1904.05677},
  year={2019}
}

```
