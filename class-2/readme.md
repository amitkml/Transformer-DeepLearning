
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
