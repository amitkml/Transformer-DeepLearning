**Team Members:**

1. Swathi Aireddy
2. Amit Kayal

The Assignment is on MNIST dataset focusing on 1. normalization techniques like Batch normalization, Group normalization and Layer normalization. 2. Regularization techniques like L1 is used and 3. It aims to learn modular code development

Hence the project consists of two files namely 1. model.py where model is defined 2. mian.py where data is loaded and model is called on this data

**Model with Group Normalization** As a first step, model is defined with group normalization

**Model with Layer Normalization** As a first step, model is defined with Layer normalization

**Model with L1 + Batch Normalization** As a first step, model is defined with L1 with Batch normalization

**Graphs** **Test/Validation Loss for all above 3 models together** On observing the graph, loss is not changing with L1. hence no improvement in performanceof the mosel. With group normalization, there is a gradual decrease in loss and finally settled to a value. With Layer normalization, initially there is a huge fluctuations in the loss change but finally it settled with a value.

[![val_loss_graph](https://user-images.githubusercontent.com/53993241/139697476-d82080b6-788b-44d3-853c-a0f148075cdc.png)](https://user-images.githubusercontent.com/53993241/139697476-d82080b6-788b-44d3-853c-a0f148075cdc.png)

**Test/Validation Accuracy for 3 models together**

On observing the graph, the val accuracy of the model with L1 and Batch normalization is just around 82% where as with out L1 the accuracy was approx. 99.4% It clearly shows that the L1 is deteriorating the accuracy of the model.

With addition of group normalization, the val accuracy of the model is 99.27%. from this we can note that batch normalization has a good impact than the group normalization.

With addition of layer normalization, the training accuaracy of the model is 99.56% and the val accuracy is 99.31% . It has a very little improvement in the performance of the model but model is overfit.

[![val_acc_graph](https://user-images.githubusercontent.com/53993241/139697561-14edea09-3f95-4b25-9bca-c53871ea5d80.png)](https://user-images.githubusercontent.com/53993241/139697561-14edea09-3f95-4b25-9bca-c53871ea5d80.png)

**10 misclassified images for model with Group Normalization** [![GN_misclassified](https://user-images.githubusercontent.com/53993241/139700148-f2bcaef7-4ee2-42c0-aad2-e0c2234d2481.png)](https://user-images.githubusercontent.com/53993241/139700148-f2bcaef7-4ee2-42c0-aad2-e0c2234d2481.png)

**10 misclassified images for model with Layer Normalization** [![LN_misclassified](https://user-images.githubusercontent.com/53993241/139700255-8d425ecb-fc10-451e-a008-309294bfa061.png)](https://user-images.githubusercontent.com/53993241/139700255-8d425ecb-fc10-451e-a008-309294bfa061.png)

**10 misclassified images for model with L1 and Batch Normalization** [![L1_BN_misclassified](https://user-images.githubusercontent.com/53993241/139700304-71078ba6-2e1a-46fd-ac94-0d74b30d08fa.png)](https://user-images.githubusercontent.com/53993241/139700304-71078ba6-2e1a-46fd-ac94-0d74b30d08fa.png)