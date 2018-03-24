# DBPN-Pytorch
Deep Back-Projection Networks for Super-Resolution (to appear in CVPR2018)
Project page: http://www.toyota-ti.ac.jp/Lab/Denshi/iim/members/muhammad.haris/projects/DBPN.html

Pretrained models can be downloaded from this link!
https://drive.google.com/drive/folders/1ahbeoEHkjxoo4NV1wReOmpoRWbl448z-?usp=sharing

It contains 4 files:
(1) DBPN_x2.pth, (2) DBPN_x4.pth, (3) DBPN_x8.pth is from the original architecture which explained in the CVPR2018 manuscript.
(4) NTIRE2018_x8.pth is used for NTIRE2018 competition (Track 1: Classic Bicubic x8)


HOW TO

----------------

#Training

    ```bash
    python main.py
    ```

#Testing

    ```bash
    python eval.py
    ```

![DBPN](http://www.toyota-ti.ac.jp/Lab/Denshi/iim/members/muhammad.haris/projects/DBPN.png)

