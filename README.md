# Параметры

Коробка размером 1x1x2 по осях X, Y, Z. Центр в (0, 0, 0).

Размер вокселя 0.01

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
