# Held-out evaluation of the learned detector (preregistration_increment11_frozen.json)

| mode | records | hours | seizures | TP | FN | FP | sensitivity | FA/h | latency median s | latency p90 s |
|---|---|---|---|---|---|---|---|---|---|---|
| offline-threshold-default | 20 | 57.8 | 30 | 28 | 2 | 410 | 0.93 | 7.09 | 5.5 | 61.9 |
| offline-ml-A | 20 | 57.8 | 30 | 28 | 2 | 374 | 0.93 | 6.47 | 1.0 | 6.6 |
| realtime-threshold-default | 20 | 57.8 | 30 | 26 | 4 | 132 | 0.87 | 2.28 | 17.5 | 57.5 |
| realtime-ml-A | 20 | 57.8 | 30 | 28 | 2 | 214 | 0.93 | 3.70 | 10.0 | 29.9 |

## Per record — offline-threshold-default

| file | seizures | TP | FN | FP | latencies s |
|---|---|---|---|---|---|
| p10/p10_Record1.edf | 1 | 1 | 0 | 3 | 102 |
| p10/p10_Record2.edf | 1 | 1 | 0 | 54 | 61 |
| p11/p11_Record1.edf | 1 | 1 | 0 | 6 | -6 |
| p11/p11_Record2.edf | 1 | 1 | 0 | 2 | -20 |
| p11/p11_Record3.edf | 4 | 4 | 0 | 2 | 11, 10, 4, 7 |
| p11/p11_Record4.edf | 1 | 1 | 0 | 15 | 125 |
| p12/p12_Record1.edf | 2 | 2 | 0 | 33 | 0, 4 |
| p12/p12_Record2.edf | 1 | 1 | 0 | 25 | 0 |
| p12/p12_Record3.edf | 3 | 3 | 0 | 13 | 10, -26, 1 |
| p13/p13_Record1.edf | 0 | 0 | 0 | 53 | — |
| p13/p13_Record2.edf | 0 | 0 | 0 | 36 | — |
| p13/p13_Record3.edf | 0 | 0 | 0 | 19 | — |
| p13/p13_Record4.edf | 2 | 2 | 0 | 44 | -9, 8 |
| p14/p14_Record1.edf | 2 | 0 | 2 | 8 | — |
| p14/p14_Record2.edf | 2 | 2 | 0 | 3 | 7, -20 |
| p14/p14_Record3.edf | 4 | 4 | 0 | 7 | -5, -8, -5, -7 |
| p15/p15_Record1.edf | 1 | 1 | 0 | 26 | 12 |
| p15/p15_Record2.edf | 1 | 1 | 0 | 13 | 7 |
| p15/p15_Record3.edf | 1 | 1 | 0 | 40 | 19 |
| p15/p15_Record4.edf | 2 | 2 | 0 | 8 | 64, 14 |

## Per record — offline-ml-A

| file | seizures | TP | FN | FP | latencies s |
|---|---|---|---|---|---|
| p10/p10_Record1.edf | 1 | 1 | 0 | 12 | -302 |
| p10/p10_Record2.edf | 1 | 1 | 0 | 23 | -56 |
| p11/p11_Record1.edf | 1 | 1 | 0 | 10 | -4 |
| p11/p11_Record2.edf | 1 | 1 | 0 | 3 | -23 |
| p11/p11_Record3.edf | 4 | 4 | 0 | 6 | 5, 11, 5, 2 |
| p11/p11_Record4.edf | 1 | 1 | 0 | 24 | 28 |
| p12/p12_Record1.edf | 2 | 2 | 0 | 47 | 2, 6 |
| p12/p12_Record2.edf | 1 | 1 | 0 | 26 | 3 |
| p12/p12_Record3.edf | 3 | 3 | 0 | 30 | 3, 5, 3 |
| p13/p13_Record1.edf | 0 | 0 | 0 | 22 | — |
| p13/p13_Record2.edf | 0 | 0 | 0 | 9 | — |
| p13/p13_Record3.edf | 0 | 0 | 0 | 7 | — |
| p13/p13_Record4.edf | 2 | 2 | 0 | 17 | -9, 8 |
| p14/p14_Record1.edf | 2 | 0 | 2 | 11 | — |
| p14/p14_Record2.edf | 2 | 2 | 0 | 12 | 4, -152 |
| p14/p14_Record3.edf | 4 | 4 | 0 | 8 | -5, -15, -2, -4 |
| p15/p15_Record1.edf | 1 | 1 | 0 | 30 | 3 |
| p15/p15_Record2.edf | 1 | 1 | 0 | 46 | -9 |
| p15/p15_Record3.edf | 1 | 1 | 0 | 15 | 0 |
| p15/p15_Record4.edf | 2 | 2 | 0 | 16 | -12, -49 |

## Per record — realtime-threshold-default

| file | seizures | TP | FN | FP | latencies s |
|---|---|---|---|---|---|
| p10/p10_Record1.edf | 1 | 1 | 0 | 5 | 116 |
| p10/p10_Record2.edf | 1 | 1 | 0 | 10 | 225 |
| p11/p11_Record1.edf | 1 | 1 | 0 | 0 | 10 |
| p11/p11_Record2.edf | 1 | 1 | 0 | 2 | -2 |
| p11/p11_Record3.edf | 4 | 4 | 0 | 1 | 18, 13, 19, 22 |
| p11/p11_Record4.edf | 1 | 0 | 1 | 10 | — |
| p12/p12_Record1.edf | 2 | 2 | 0 | 13 | 14, 21 |
| p12/p12_Record2.edf | 1 | 1 | 0 | 13 | 13 |
| p12/p12_Record3.edf | 3 | 3 | 0 | 6 | 17, 16, 14 |
| p13/p13_Record1.edf | 0 | 0 | 0 | 12 | — |
| p13/p13_Record2.edf | 0 | 0 | 0 | 8 | — |
| p13/p13_Record3.edf | 0 | 0 | 0 | 11 | — |
| p13/p13_Record4.edf | 2 | 2 | 0 | 9 | 5, 24 |
| p14/p14_Record1.edf | 2 | 0 | 2 | 4 | — |
| p14/p14_Record2.edf | 2 | 2 | 0 | 1 | 27, -1 |
| p14/p14_Record3.edf | 4 | 3 | 1 | 5 | 9, 6, 6 |
| p15/p15_Record1.edf | 1 | 1 | 0 | 5 | 27 |
| p15/p15_Record2.edf | 1 | 1 | 0 | 9 | 23 |
| p15/p15_Record3.edf | 1 | 1 | 0 | 2 | 34 |
| p15/p15_Record4.edf | 2 | 2 | 0 | 6 | 79, 36 |

## Per record — realtime-ml-A

| file | seizures | TP | FN | FP | latencies s |
|---|---|---|---|---|---|
| p10/p10_Record1.edf | 1 | 1 | 0 | 6 | -20 |
| p10/p10_Record2.edf | 1 | 1 | 0 | 24 | 73 |
| p11/p11_Record1.edf | 1 | 1 | 0 | 4 | 2 |
| p11/p11_Record2.edf | 1 | 1 | 0 | 1 | -6 |
| p11/p11_Record3.edf | 4 | 4 | 0 | 0 | 18, 18, 18, 26 |
| p11/p11_Record4.edf | 1 | 1 | 0 | 11 | 123 |
| p12/p12_Record1.edf | 2 | 2 | 0 | 41 | 11, 13 |
| p12/p12_Record2.edf | 1 | 1 | 0 | 13 | 10 |
| p12/p12_Record3.edf | 3 | 3 | 0 | 15 | 9, 12, 10 |
| p13/p13_Record1.edf | 0 | 0 | 0 | 5 | — |
| p13/p13_Record2.edf | 0 | 0 | 0 | 3 | — |
| p13/p13_Record3.edf | 0 | 0 | 0 | 1 | — |
| p13/p13_Record4.edf | 2 | 2 | 0 | 8 | 9, 15 |
| p14/p14_Record1.edf | 2 | 0 | 2 | 4 | — |
| p14/p14_Record2.edf | 2 | 2 | 0 | 7 | 12, -34 |
| p14/p14_Record3.edf | 4 | 4 | 0 | 7 | 1, -4, 6, 5 |
| p15/p15_Record1.edf | 1 | 1 | 0 | 19 | 19 |
| p15/p15_Record2.edf | 1 | 1 | 0 | 24 | -3 |
| p15/p15_Record3.edf | 1 | 1 | 0 | 13 | 8 |
| p15/p15_Record4.edf | 2 | 2 | 0 | 8 | 39, -23 |
