# Deep neural network test with flowers recogition

## Install and run

```sh
git clone https://github.com/illiatdesdindes/DeepNeuralNetworkFlowers.git
```

download dataset from [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition) and extract content in `flowers/` directoy

```
flowers
|- daisy
   |- ...jpg
   |- ...jpg
   ...
|- dandelion
   |- ...jpg
   ...
...
```
then
```sh
dotnet restore
dotnet run
```
and wait...

if everything goes well the model should be trained, and then tested against the images inside if the `test` folder

```sh
Image: tulip.jpg, Predicted label: tulip
Image: sunflower.jpg, Predicted label: sunflower
Image: dandelion.jpg, Predicted label: dandelion
Image: rose.jpg, Predicted label: rose
Image: daisy.jpg, Predicted label: daisy
```



### Acknowledgment

- Inspired from [Creating a Deep Neural Network in ML.NET - Microsoft.ML.Vision Update](youtube.com/watch?v=ppRauvf6uCs)
- dataset from [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)