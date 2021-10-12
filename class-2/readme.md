
## Assignment

- Write a neural network that can:
    - take 2 inputs:
      - an image from the MNIST dataset (say 5), and
      - a random number between 0 and 9, (say 7)
    - and gives two outputs:
      - the "number" that was represented by the MNIST image (predict 5), and
      - the "sum" of this number with the random number and the input image to  the network (predict 5 + 7 = 12)
      ![im](https://canvas.instructure.com/courses/2734471/files/155148058/preview)
- you can mix fully connected layers and convolution layers
- you can use one-hot encoding to represent the random number input as well as the "summed" output.
    - Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
    - Sum (13) can be represented as:
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010
- Your code MUST be:
  - well documented (via readme file on GitHub and comments in the code)
  - must mention the data representation
  - must mention your data generation strategy (basically the class/method you are using for random number generation)
  - must mention how you have combined the two inputs (basically which layer you are combining)
  - must mention how you are evaluating your results 
  - must mention "what" results you finally got and how did you evaluate your results
  - must mention what loss function you picked and why!
  - training MUST happen on the GPU

## Training Log

### Training and Test Evaluation Chart

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/class-2/Test_result.png?raw=true)
![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/class-2/Training_result.png?raw=true)

### Log

/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:66: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Avg. loss: 2.3072, Accuracy: 1110/10000 (11%)

Train Epoch: 1 [0/60000 (0%)]	Loss: 2.308383
Train Epoch: 1 [640/60000 (1%)]	Loss: 2.310192
Train Epoch: 1 [1280/60000 (2%)]	Loss: 2.293060
Train Epoch: 1 [1920/60000 (3%)]	Loss: 2.274329
Train Epoch: 1 [2560/60000 (4%)]	Loss: 2.288037
Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.285798
Train Epoch: 1 [3840/60000 (6%)]	Loss: 2.263713
Train Epoch: 1 [4480/60000 (7%)]	Loss: 2.251025
Train Epoch: 1 [5120/60000 (9%)]	Loss: 2.253286
Train Epoch: 1 [5760/60000 (10%)]	Loss: 2.204738
Train Epoch: 1 [6400/60000 (11%)]	Loss: 2.133570
Train Epoch: 1 [7040/60000 (12%)]	Loss: 2.046288
Train Epoch: 1 [7680/60000 (13%)]	Loss: 1.618983
Train Epoch: 1 [8320/60000 (14%)]	Loss: 1.453925
Train Epoch: 1 [8960/60000 (15%)]	Loss: 1.147109
Train Epoch: 1 [9600/60000 (16%)]	Loss: 0.898455
Train Epoch: 1 [10240/60000 (17%)]	Loss: 0.856337
Train Epoch: 1 [10880/60000 (18%)]	Loss: 0.918861
Train Epoch: 1 [11520/60000 (19%)]	Loss: 0.651475
Train Epoch: 1 [12160/60000 (20%)]	Loss: 0.897561
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.579376
Train Epoch: 1 [13440/60000 (22%)]	Loss: 0.851848
Train Epoch: 1 [14080/60000 (23%)]	Loss: 0.721473
Train Epoch: 1 [14720/60000 (25%)]	Loss: 0.530455
Train Epoch: 1 [15360/60000 (26%)]	Loss: 0.601496
Train Epoch: 1 [16000/60000 (27%)]	Loss: 0.905740
Train Epoch: 1 [16640/60000 (28%)]	Loss: 0.749485
Train Epoch: 1 [17280/60000 (29%)]	Loss: 0.854803
Train Epoch: 1 [17920/60000 (30%)]	Loss: 0.665698
Train Epoch: 1 [18560/60000 (31%)]	Loss: 0.677264
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.431477
Train Epoch: 1 [19840/60000 (33%)]	Loss: 0.764341
Train Epoch: 1 [20480/60000 (34%)]	Loss: 0.693712
Train Epoch: 1 [21120/60000 (35%)]	Loss: 0.750022
Train Epoch: 1 [21760/60000 (36%)]	Loss: 0.749102
Train Epoch: 1 [22400/60000 (37%)]	Loss: 0.533354
Train Epoch: 1 [23040/60000 (38%)]	Loss: 0.451555
Train Epoch: 1 [23680/60000 (39%)]	Loss: 0.424665
Train Epoch: 1 [24320/60000 (41%)]	Loss: 0.658196
Train Epoch: 1 [24960/60000 (42%)]	Loss: 0.614043
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.719853
Train Epoch: 1 [26240/60000 (44%)]	Loss: 0.330175
Train Epoch: 1 [26880/60000 (45%)]	Loss: 0.402121
Train Epoch: 1 [27520/60000 (46%)]	Loss: 0.437398
Train Epoch: 1 [28160/60000 (47%)]	Loss: 0.367952
Train Epoch: 1 [28800/60000 (48%)]	Loss: 0.433275
Train Epoch: 1 [29440/60000 (49%)]	Loss: 0.660267
Train Epoch: 1 [30080/60000 (50%)]	Loss: 0.470273
Train Epoch: 1 [30720/60000 (51%)]	Loss: 0.741625
Train Epoch: 1 [31360/60000 (52%)]	Loss: 0.676518
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.282849
Train Epoch: 1 [32640/60000 (54%)]	Loss: 0.282628
Train Epoch: 1 [33280/60000 (55%)]	Loss: 0.500469
Train Epoch: 1 [33920/60000 (57%)]	Loss: 0.243350
Train Epoch: 1 [34560/60000 (58%)]	Loss: 0.313930
Train Epoch: 1 [35200/60000 (59%)]	Loss: 0.891542
Train Epoch: 1 [35840/60000 (60%)]	Loss: 0.348013
Train Epoch: 1 [36480/60000 (61%)]	Loss: 0.660020
Train Epoch: 1 [37120/60000 (62%)]	Loss: 0.536946
Train Epoch: 1 [37760/60000 (63%)]	Loss: 0.471192
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.391710
Train Epoch: 1 [39040/60000 (65%)]	Loss: 0.367860
Train Epoch: 1 [39680/60000 (66%)]	Loss: 0.326909
Train Epoch: 1 [40320/60000 (67%)]	Loss: 0.464428
Train Epoch: 1 [40960/60000 (68%)]	Loss: 0.258677
Train Epoch: 1 [41600/60000 (69%)]	Loss: 0.239572
Train Epoch: 1 [42240/60000 (70%)]	Loss: 0.216849
Train Epoch: 1 [42880/60000 (71%)]	Loss: 0.417677
Train Epoch: 1 [43520/60000 (72%)]	Loss: 0.382677
Train Epoch: 1 [44160/60000 (74%)]	Loss: 0.294750
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.280540
Train Epoch: 1 [45440/60000 (76%)]	Loss: 0.237570
Train Epoch: 1 [46080/60000 (77%)]	Loss: 0.295069
Train Epoch: 1 [46720/60000 (78%)]	Loss: 0.544531
Train Epoch: 1 [47360/60000 (79%)]	Loss: 0.304629
Train Epoch: 1 [48000/60000 (80%)]	Loss: 0.301356
Train Epoch: 1 [48640/60000 (81%)]	Loss: 0.283874
Train Epoch: 1 [49280/60000 (82%)]	Loss: 0.312440
Train Epoch: 1 [49920/60000 (83%)]	Loss: 0.248730
Train Epoch: 1 [50560/60000 (84%)]	Loss: 0.227366
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.213443
Train Epoch: 1 [51840/60000 (86%)]	Loss: 0.198088
Train Epoch: 1 [52480/60000 (87%)]	Loss: 0.455670
Train Epoch: 1 [53120/60000 (88%)]	Loss: 0.180053
Train Epoch: 1 [53760/60000 (90%)]	Loss: 0.311541
Train Epoch: 1 [54400/60000 (91%)]	Loss: 0.237455
Train Epoch: 1 [55040/60000 (92%)]	Loss: 0.416413
Train Epoch: 1 [55680/60000 (93%)]	Loss: 0.264704
Train Epoch: 1 [56320/60000 (94%)]	Loss: 0.182968
Train Epoch: 1 [56960/60000 (95%)]	Loss: 0.455384
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.254990
Train Epoch: 1 [58240/60000 (97%)]	Loss: 0.382746
Train Epoch: 1 [58880/60000 (98%)]	Loss: 0.501192
Train Epoch: 1 [59520/60000 (99%)]	Loss: 0.173629

Test set: Avg. loss: 0.1574, Accuracy: 9508/10000 (95%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 0.232186
Train Epoch: 2 [640/60000 (1%)]	Loss: 0.229126
Train Epoch: 2 [1280/60000 (2%)]	Loss: 0.363159
Train Epoch: 2 [1920/60000 (3%)]	Loss: 0.199877
Train Epoch: 2 [2560/60000 (4%)]	Loss: 0.235210
Train Epoch: 2 [3200/60000 (5%)]	Loss: 0.321126
Train Epoch: 2 [3840/60000 (6%)]	Loss: 0.443914
Train Epoch: 2 [4480/60000 (7%)]	Loss: 0.181147
Train Epoch: 2 [5120/60000 (9%)]	Loss: 0.330016
Train Epoch: 2 [5760/60000 (10%)]	Loss: 0.282348
Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.161066
Train Epoch: 2 [7040/60000 (12%)]	Loss: 0.105714
Train Epoch: 2 [7680/60000 (13%)]	Loss: 0.280961
Train Epoch: 2 [8320/60000 (14%)]	Loss: 0.262988
Train Epoch: 2 [8960/60000 (15%)]	Loss: 0.244343
Train Epoch: 2 [9600/60000 (16%)]	Loss: 0.280135
Train Epoch: 2 [10240/60000 (17%)]	Loss: 0.138108
Train Epoch: 2 [10880/60000 (18%)]	Loss: 0.290898
Train Epoch: 2 [11520/60000 (19%)]	Loss: 0.183653
Train Epoch: 2 [12160/60000 (20%)]	Loss: 0.229720
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.309133
Train Epoch: 2 [13440/60000 (22%)]	Loss: 0.350202
Train Epoch: 2 [14080/60000 (23%)]	Loss: 0.384396
Train Epoch: 2 [14720/60000 (25%)]	Loss: 0.196918
Train Epoch: 2 [15360/60000 (26%)]	Loss: 0.152126
Train Epoch: 2 [16000/60000 (27%)]	Loss: 0.293374
Train Epoch: 2 [16640/60000 (28%)]	Loss: 0.171283
Train Epoch: 2 [17280/60000 (29%)]	Loss: 0.397023
Train Epoch: 2 [17920/60000 (30%)]	Loss: 0.137691
Train Epoch: 2 [18560/60000 (31%)]	Loss: 0.354693
Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.280696
Train Epoch: 2 [19840/60000 (33%)]	Loss: 0.258504
Train Epoch: 2 [20480/60000 (34%)]	Loss: 0.134152
Train Epoch: 2 [21120/60000 (35%)]	Loss: 0.269836
Train Epoch: 2 [21760/60000 (36%)]	Loss: 0.295002
Train Epoch: 2 [22400/60000 (37%)]	Loss: 0.366046
Train Epoch: 2 [23040/60000 (38%)]	Loss: 0.288011
Train Epoch: 2 [23680/60000 (39%)]	Loss: 0.169313
Train Epoch: 2 [24320/60000 (41%)]	Loss: 0.239494
Train Epoch: 2 [24960/60000 (42%)]	Loss: 0.185455
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.194577
Train Epoch: 2 [26240/60000 (44%)]	Loss: 0.155594
Train Epoch: 2 [26880/60000 (45%)]	Loss: 0.201440
Train Epoch: 2 [27520/60000 (46%)]	Loss: 0.205766
Train Epoch: 2 [28160/60000 (47%)]	Loss: 0.134400
Train Epoch: 2 [28800/60000 (48%)]	Loss: 0.215424
Train Epoch: 2 [29440/60000 (49%)]	Loss: 0.217165
Train Epoch: 2 [30080/60000 (50%)]	Loss: 0.210960
Train Epoch: 2 [30720/60000 (51%)]	Loss: 0.145822
Train Epoch: 2 [31360/60000 (52%)]	Loss: 0.075516
Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.156566
Train Epoch: 2 [32640/60000 (54%)]	Loss: 0.255787
Train Epoch: 2 [33280/60000 (55%)]	Loss: 0.253503
Train Epoch: 2 [33920/60000 (57%)]	Loss: 0.354456
Train Epoch: 2 [34560/60000 (58%)]	Loss: 0.304528
Train Epoch: 2 [35200/60000 (59%)]	Loss: 0.205827
Train Epoch: 2 [35840/60000 (60%)]	Loss: 0.199398
Train Epoch: 2 [36480/60000 (61%)]	Loss: 0.146250
Train Epoch: 2 [37120/60000 (62%)]	Loss: 0.111353
Train Epoch: 2 [37760/60000 (63%)]	Loss: 0.165628
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.271851
Train Epoch: 2 [39040/60000 (65%)]	Loss: 0.180936
Train Epoch: 2 [39680/60000 (66%)]	Loss: 0.135094
Train Epoch: 2 [40320/60000 (67%)]	Loss: 0.230814
Train Epoch: 2 [40960/60000 (68%)]	Loss: 0.209723
Train Epoch: 2 [41600/60000 (69%)]	Loss: 0.065978
Train Epoch: 2 [42240/60000 (70%)]	Loss: 0.249991
Train Epoch: 2 [42880/60000 (71%)]	Loss: 0.233860
Train Epoch: 2 [43520/60000 (72%)]	Loss: 0.234831
Train Epoch: 2 [44160/60000 (74%)]	Loss: 0.099989
Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.247536
Train Epoch: 2 [45440/60000 (76%)]	Loss: 0.325648
Train Epoch: 2 [46080/60000 (77%)]	Loss: 0.200508
Train Epoch: 2 [46720/60000 (78%)]	Loss: 0.107369
Train Epoch: 2 [47360/60000 (79%)]	Loss: 0.204334
Train Epoch: 2 [48000/60000 (80%)]	Loss: 0.082969
Train Epoch: 2 [48640/60000 (81%)]	Loss: 0.150873
Train Epoch: 2 [49280/60000 (82%)]	Loss: 0.152246
Train Epoch: 2 [49920/60000 (83%)]	Loss: 0.293441
Train Epoch: 2 [50560/60000 (84%)]	Loss: 0.054647
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.233656
Train Epoch: 2 [51840/60000 (86%)]	Loss: 0.113822
Train Epoch: 2 [52480/60000 (87%)]	Loss: 0.182256
Train Epoch: 2 [53120/60000 (88%)]	Loss: 0.236332
Train Epoch: 2 [53760/60000 (90%)]	Loss: 0.137302
Train Epoch: 2 [54400/60000 (91%)]	Loss: 0.277315
Train Epoch: 2 [55040/60000 (92%)]	Loss: 0.167667
Train Epoch: 2 [55680/60000 (93%)]	Loss: 0.116008
Train Epoch: 2 [56320/60000 (94%)]	Loss: 0.089791
Train Epoch: 2 [56960/60000 (95%)]	Loss: 0.118934
Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.217101
Train Epoch: 2 [58240/60000 (97%)]	Loss: 0.100815
Train Epoch: 2 [58880/60000 (98%)]	Loss: 0.175545
Train Epoch: 2 [59520/60000 (99%)]	Loss: 0.132878

Test set: Avg. loss: 0.0755, Accuracy: 9755/10000 (98%)

Train Epoch: 3 [0/60000 (0%)]	Loss: 0.099359
Train Epoch: 3 [640/60000 (1%)]	Loss: 0.119976
Train Epoch: 3 [1280/60000 (2%)]	Loss: 0.109336
Train Epoch: 3 [1920/60000 (3%)]	Loss: 0.310894
Train Epoch: 3 [2560/60000 (4%)]	Loss: 0.189996
Train Epoch: 3 [3200/60000 (5%)]	Loss: 0.087060
Train Epoch: 3 [3840/60000 (6%)]	Loss: 0.168573
Train Epoch: 3 [4480/60000 (7%)]	Loss: 0.109686
Train Epoch: 3 [5120/60000 (9%)]	Loss: 0.185162
Train Epoch: 3 [5760/60000 (10%)]	Loss: 0.294218
Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.254325
Train Epoch: 3 [7040/60000 (12%)]	Loss: 0.153712
Train Epoch: 3 [7680/60000 (13%)]	Loss: 0.360580
Train Epoch: 3 [8320/60000 (14%)]	Loss: 0.158423
Train Epoch: 3 [8960/60000 (15%)]	Loss: 0.185303
Train Epoch: 3 [9600/60000 (16%)]	Loss: 0.112644
Train Epoch: 3 [10240/60000 (17%)]	Loss: 0.148236
Train Epoch: 3 [10880/60000 (18%)]	Loss: 0.101123
Train Epoch: 3 [11520/60000 (19%)]	Loss: 0.090055
Train Epoch: 3 [12160/60000 (20%)]	Loss: 0.224404
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.377445
Train Epoch: 3 [13440/60000 (22%)]	Loss: 0.192831
Train Epoch: 3 [14080/60000 (23%)]	Loss: 0.171705
Train Epoch: 3 [14720/60000 (25%)]	Loss: 0.154525
Train Epoch: 3 [15360/60000 (26%)]	Loss: 0.174377
Train Epoch: 3 [16000/60000 (27%)]	Loss: 0.229947
Train Epoch: 3 [16640/60000 (28%)]	Loss: 0.254559
Train Epoch: 3 [17280/60000 (29%)]	Loss: 0.123267
Train Epoch: 3 [17920/60000 (30%)]	Loss: 0.232038
Train Epoch: 3 [18560/60000 (31%)]	Loss: 0.364219
Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.284862
Train Epoch: 3 [19840/60000 (33%)]	Loss: 0.111114
Train Epoch: 3 [20480/60000 (34%)]	Loss: 0.142210
Train Epoch: 3 [21120/60000 (35%)]	Loss: 0.202080
Train Epoch: 3 [21760/60000 (36%)]	Loss: 0.140229
Train Epoch: 3 [22400/60000 (37%)]	Loss: 0.195704
Train Epoch: 3 [23040/60000 (38%)]	Loss: 0.331430
Train Epoch: 3 [23680/60000 (39%)]	Loss: 0.133644
Train Epoch: 3 [24320/60000 (41%)]	Loss: 0.101400
Train Epoch: 3 [24960/60000 (42%)]	Loss: 0.140223
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.275296
Train Epoch: 3 [26240/60000 (44%)]	Loss: 0.090543
Train Epoch: 3 [26880/60000 (45%)]	Loss: 0.140876
Train Epoch: 3 [27520/60000 (46%)]	Loss: 0.201091
Train Epoch: 3 [28160/60000 (47%)]	Loss: 0.111967
Train Epoch: 3 [28800/60000 (48%)]	Loss: 0.081309
Train Epoch: 3 [29440/60000 (49%)]	Loss: 0.099816
Train Epoch: 3 [30080/60000 (50%)]	Loss: 0.065747
Train Epoch: 3 [30720/60000 (51%)]	Loss: 0.111002
Train Epoch: 3 [31360/60000 (52%)]	Loss: 0.214689
Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.055765
Train Epoch: 3 [32640/60000 (54%)]	Loss: 0.098562
Train Epoch: 3 [33280/60000 (55%)]	Loss: 0.249470
Train Epoch: 3 [33920/60000 (57%)]	Loss: 0.090898
Train Epoch: 3 [34560/60000 (58%)]	Loss: 0.162890
Train Epoch: 3 [35200/60000 (59%)]	Loss: 0.188552
Train Epoch: 3 [35840/60000 (60%)]	Loss: 0.195051
Train Epoch: 3 [36480/60000 (61%)]	Loss: 0.110803
Train Epoch: 3 [37120/60000 (62%)]	Loss: 0.214891
Train Epoch: 3 [37760/60000 (63%)]	Loss: 0.350683
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.159218
Train Epoch: 3 [39040/60000 (65%)]	Loss: 0.213174
Train Epoch: 3 [39680/60000 (66%)]	Loss: 0.164575
Train Epoch: 3 [40320/60000 (67%)]	Loss: 0.138938
Train Epoch: 3 [40960/60000 (68%)]	Loss: 0.046750
Train Epoch: 3 [41600/60000 (69%)]	Loss: 0.375724
Train Epoch: 3 [42240/60000 (70%)]	Loss: 0.125548
Train Epoch: 3 [42880/60000 (71%)]	Loss: 0.230186
Train Epoch: 3 [43520/60000 (72%)]	Loss: 0.087542
Train Epoch: 3 [44160/60000 (74%)]	Loss: 0.123750
Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.197126
Train Epoch: 3 [45440/60000 (76%)]	Loss: 0.095662
Train Epoch: 3 [46080/60000 (77%)]	Loss: 0.277210
Train Epoch: 3 [46720/60000 (78%)]	Loss: 0.128528
Train Epoch: 3 [47360/60000 (79%)]	Loss: 0.165587
Train Epoch: 3 [48000/60000 (80%)]	Loss: 0.145595
Train Epoch: 3 [48640/60000 (81%)]	Loss: 0.109647
Train Epoch: 3 [49280/60000 (82%)]	Loss: 0.166814
Train Epoch: 3 [49920/60000 (83%)]	Loss: 0.153422
Train Epoch: 3 [50560/60000 (84%)]	Loss: 0.056184
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.067029
Train Epoch: 3 [51840/60000 (86%)]	Loss: 0.259974
Train Epoch: 3 [52480/60000 (87%)]	Loss: 0.161985
Train Epoch: 3 [53120/60000 (88%)]	Loss: 0.106582
Train Epoch: 3 [53760/60000 (90%)]	Loss: 0.105688
Train Epoch: 3 [54400/60000 (91%)]	Loss: 0.124062
Train Epoch: 3 [55040/60000 (92%)]	Loss: 0.155764
Train Epoch: 3 [55680/60000 (93%)]	Loss: 0.256857
Train Epoch: 3 [56320/60000 (94%)]	Loss: 0.168465
Train Epoch: 3 [56960/60000 (95%)]	Loss: 0.192001
Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.159685
Train Epoch: 3 [58240/60000 (97%)]	Loss: 0.129784
Train Epoch: 3 [58880/60000 (98%)]	Loss: 0.268377
Train Epoch: 3 [59520/60000 (99%)]	Loss: 0.155245

Test set: Avg. loss: 0.0536, Accuracy: 9824/10000 (98%)
