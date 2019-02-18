# Object-Detection-on-2-digit-MNIST
Object Detection and Classification on 2-Digit MNIST Dataset. As you can see in results.png, the currently trained model (ckpt) classifys both digits correctly 96% of the time with an iou on the predicted bounding boxes of 0.91. 
#
The compressed dataset is located in the data folder and contains 55000 64x64 grayscale images of two handwritten digits randomly orientated inside the image. Also inside is the coresponding labels and bounding boxes of each image.
#
To train a new model, make train=True on line 8 of main.py.

## Some examples

![ui_v1.0](https://github.com/cdamore/Object-Detection-on-2-digit-MNIST/blob/master/examples/ex1.jpg?raw=true)
![ui_v1.0](https://github.com/cdamore/Object-Detection-on-2-digit-MNIST/blob/master/examples/ex2.jpg?raw=true)
![ui_v1.0](https://github.com/cdamore/Object-Detection-on-2-digit-MNIST/blob/master/examples/ex3.jpg?raw=true)
![ui_v1.0](https://github.com/cdamore/Object-Detection-on-2-digit-MNIST/blob/master/examples/ex4.jpg?raw=true)
![ui_v1.0](https://github.com/cdamore/Object-Detection-on-2-digit-MNIST/blob/master/examples/ex5.jpg?raw=true)
