# cli

```shell
python cli.py data --num 100 --vox-size 0.02
```

# Параметры

Коробка размером 1x1x2 по осях X, Y, Z. Центр в (0, 0, 0).

Размер вокселя 0.01

Поворот делается по оси (0, 0, 1)

# Результаты

|  Поворот | ICP                | Расстояние |
|---------:|--------------------|-----------:|
|        0 |                    |      0.005 |
|        0 | :heavy_check_mark: |      0.005 |
| 10&#xb0; |                    |      0.011 |
| 10&#xb0; | :heavy_check_mark: |      0.011 |

## Без поворота, без ICP

![box_1_1_2_r0](static/box_1_1_2_r0.png)

## Без поворота, с ICP

![box_1_1_2_r0_icp](static/box_1_1_2_r0_icp.png)

## Поворот 10&#xb0;, без ICP

![box_1_1_2_r10](static/box_1_1_2_r10.png)

## Поворот 10&#xb0;, с ICP

![box_1_1_2_r10_icp](static/box_1_1_2_r10_icp.png)

# Параметры

![plane](https://ars.els-cdn.com/content/image/1-s2.0-S2214657116300065-gr004_lrg.jpg)

Размеры 4.8x4.8x0.8.
Размер вокселя - 0.02.

# Результаты

|  Поворот | ICP                | Расстояние |
|---------:|--------------------|-----------:|
|        0 |                    |      0.020 |
|        0 | :heavy_check_mark: |      0.020 |
| 10&#xb0; |                    |      0.022 |
| 10&#xb0; | :heavy_check_mark: |      0.022 |

![plane_r0](static/plane_r0.png)

![plane_r0_icp](static/plane_r0_icp.png)

![plane_r10](static/plane_r10.png)

![plane_r10_icp](static/plane_r10_icp.png)
