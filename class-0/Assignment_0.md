# Assignment 1B

## Assignee Details



- **Name**: Amit Kayal
- **Email Address**: [amitkayal@outlook.com](mailto:amitkayal@outlook.com)
- **Batch**: Saturday 11.15 A.M

## Assignment Solution



### What are Channels and Kernels (according to EVA)?

Filter/Kernel is a matrix of some values with defined filter size. Our expectation is that weights (or values) in the convolution filter multiplied with corresponding input image pixels will gives us a value that best represents those image pixels and thus will help us to extract key features from the input image. Filter size is one of the hyper-parameters and our choice varies between higher and smaller values. 

From CNN perspective, Kernel=Filter=Feature Extractor

Kernels are called similar feature extractor.

### Why should we only (well mostly) use 3x3 Kernels?

ne of the commercial reason for using 3x3 filter is that our current GPU's(NVDIA, Intel etc) are optimized heavily for such filter. Also note that, all kernel are of Odd size and we generally don't use even size kernel. This is because am even number does not have concept of mirror and so we can't talk of things are in right and things are in left.

It is always number of parameters which are less with 3x3 filter. Any filter like 5x5 or 7x7 can be represented by a number of 3x3. Lets see how? 

**Scenario:  One(Channel=1) 5x5 filter**

- Input Size: 5x5 
- Output size: 1x1
- No of parameters: 25

**Scenario: One(Channel=1) 3x3 filter**

- Input Size: 5x5 
- Output after 1st convolution: 3x3
- Output after 2nd convolution: 1x1
- Final Output: 1x1
- No of parameters=18

Above example shows how a 5x5 filter can be represented by 2 no of 3x3 filter and also it reduces no parameters to be learning. But here no of convolution operation is twice. The receptive field of the 5x5 kernel can be covered by two 3x3 kernels.  Stacking small filters gives us more discriminative power via the non-linearity and via the composition of abstraction. 

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

Here is the Input and Output of the convolution shared below. 3x3 kernel with stride 1 has been used here. No of convolution is 100.

| Layer | Input | Output |
| ----- | ----- | ------ |
| 1     | 199   | 197    |
| 2     | 197   | 195    |
| 3     | 195   | 193    |
| 4     | 193   | 191    |
| 5     | 191   | 189    |
| 6     | 189   | 187    |
| 7     | 187   | 185    |
| 8     | 185   | 183    |
| 9     | 183   | 181    |
| 10    | 181   | 179    |
| 11    | 179   | 177    |
| 12    | 177   | 175    |
| 13    | 175   | 173    |
| 14    | 173   | 171    |
| 15    | 171   | 169    |
| 16    | 169   | 167    |
| 17    | 167   | 165    |
| 18    | 165   | 163    |
| 19    | 163   | 161    |
| 20    | 161   | 159    |
| 21    | 159   | 157    |
| 22    | 157   | 155    |
| 23    | 155   | 153    |
| 24    | 153   | 151    |
| 25    | 151   | 149    |
| 26    | 149   | 147    |
| 27    | 147   | 145    |
| 28    | 145   | 143    |
| 29    | 143   | 141    |
| 30    | 141   | 139    |
| 31    | 139   | 137    |
| 32    | 137   | 135    |
| 33    | 135   | 133    |
| 34    | 133   | 131    |
| 35    | 131   | 129    |
| 36    | 129   | 127    |
| 37    | 127   | 125    |
| 38    | 125   | 123    |
| 39    | 123   | 121    |
| 40    | 121   | 119    |
| 41    | 119   | 117    |
| 42    | 117   | 115    |
| 43    | 115   | 113    |
| 44    | 113   | 111    |
| 45    | 111   | 109    |
| 46    | 109   | 107    |
| 47    | 107   | 105    |
| 48    | 105   | 103    |
| 49    | 103   | 101    |
| 50    | 101   | 99     |
| 51    | 99    | 97     |
| 52    | 97    | 95     |
| 53    | 95    | 93     |
| 54    | 93    | 91     |
| 55    | 91    | 89     |
| 56    | 89    | 87     |
| 57    | 87    | 85     |
| 58    | 85    | 83     |
| 59    | 83    | 81     |
| 60    | 81    | 79     |
| 61    | 79    | 77     |
| 62    | 77    | 75     |
| 63    | 75    | 73     |
| 64    | 73    | 71     |
| 65    | 71    | 69     |
| 66    | 69    | 67     |
| 67    | 67    | 65     |
| 68    | 65    | 63     |
| 69    | 63    | 61     |
| 70    | 61    | 59     |
| 71    | 59    | 57     |
| 72    | 57    | 55     |
| 73    | 55    | 53     |
| 74    | 53    | 51     |
| 75    | 51    | 49     |
| 76    | 49    | 47     |
| 77    | 47    | 45     |
| 78    | 45    | 43     |
| 79    | 43    | 41     |
| 80    | 41    | 39     |
| 81    | 39    | 37     |
| 82    | 37    | 35     |
| 83    | 35    | 33     |
| 84    | 33    | 31     |
| 85    | 31    | 29     |
| 86    | 29    | 27     |
| 87    | 27    | 25     |
| 88    | 25    | 23     |
| 89    | 23    | 21     |
| 90    | 21    | 19     |
| 91    | 19    | 17     |
| 92    | 17    | 15     |
| 93    | 15    | 13     |
| 94    | 13    | 11     |
| 95    | 11    | 9      |
| 96    | 9     | 7      |
| 97    | 7     | 5      |
| 98    | 5     | 3      |
| 99    | 3     | 1      |

### How are kernels initialized? 

Kernels have different way to initialize and most frequently we initialize randomly.  Random initlaize ensures network is not biased one. too-large initialization leads to exploding gradients while too-small initialization leads to vanishing gradients.

Here are few different way we can initialize kernel.

- **Zeros**: Initializer that generates tensors initialized to 0.
- **Ones**: Initializer that generates tensors initialized to 1.
- **Constant**: Initializer that generates tensors initialized to a constant value.
- **RandomNormal**: Initializer that generates tensors with a normal distribution.
- **RandomUniform**: Initializer that generates tensors with a uniform distribution.
- **TruncatedNormal**: Initializer that generates a truncated normal distribution.
- **VarianceScaling**: Initializer capable of adapting its scale to the shape of weights.
- **Orthogonal**: Initializer that generates a random orthogonal matrix.
- **Identity**: Initializer that generates the identity matrix.
- **lecun_uniform**: LeCun uniform initializer.
- **glorot_normal**: Glorot normal initializer, also called Xavier normal initializer.
- **glorot_uniform**: Glorot uniform initializer, also called Xavier uniform initializer.
- **he_normal**: He normal initializer.
- **lecun_normal**: LeCun normal initializer.
- **he_uniform**: He uniform variance scaling initializer.

### What happens during the training of a DNN?

Network produce desired output and then loss function gets triggered to find out difference between desired and predicted output which then passes through backpropagation to change the kernel weights further to reduce the loss. Such process goes on still we can achieve into minimum or stable loss.