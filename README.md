# AML_Project_Perez_Rodriguez

# Kvasir_Final_Project-perez-rodriguez

![resultados_cualitativo](https://github.com/IBIO4490/Kvasir_Final_Project-perez-rodriguez/blob/master/images/resultados_cualitativo.jpg)

Retinanet model with a Resnet-101 backbone trained using a balanced focal loss, data upsampling and data augmentation to detect small bowel abnormalities on images from capsule endoscopy videos from the Kvasir Capsule Endoscopy dataset. 

Some qualitative results are shown in this repo, where green boxes are the predictions from the model and blue boxes are the annotations of the dataset made by experts.

# Results on the test split

* Quantitatively, our model achieves the following official COCO detection metrics:

```
Method	mIoU	IoU	mcIoU	Instrument Categories						
				BF	PF	LND	VS/SI	GR/CA	MCS	UP
MATIS RFrame (Ours)	82.96	78.69	48.07	85.01	37.53	46.14	64.23	0.00	93.47	10.10
![image](https://user-images.githubusercontent.com/98598743/204051192-64b60d87-cebb-4245-b0c1-41e7782c6cdd.png)

 ```
 * Metrics for each class:
 ```
 Angiectasia    : AP --> 0.572 - AR --> 0.653
 Blood - fresh  : AP --> 0.835 - AR --> 0.879
 Erosion        : AP --> 0.474 - AR --> 0.537
 Erythema       : AP --> 0.663 - AR --> 0.700
 Foreign Body   : AP --> 0.695 - AR --> 0.748
 Lymphangiectasia: AP --> 0.677- AR --> 0.727
 Ulcer          : AP --> 0.626 - AR --> 0.697
 Polyp          : AP --> 0.748 - AR --> 0.771
```


# Environment and installation
1. Clone this repository 
2. Create a new environment
```
 $ conda create --name env --file spec-file.txt
 $ conda install --name env --file spec-file.txt
 $ conda activate env
```
3. Install required packages 
```
 $ pip install opencv-python
 $ pip install matplotlib
 $ pip install pycocotools
 $ pip install requests
 $ pip install scikit-image
 
```
4. Soft link to the dataset and model weights
```
 $ ln -s /media/user_home0/aperezr20/Vision/Proyecto/pytorch-retinanet/Kvasir_FinalData Kvasir_FinalData
 
```

5. Create the folder in which results will be saved (evaluation of the test split metrics and qualitative demo results)
```
 $ mkdir RESULTS
 
```
# Usage

* Evaluation of the model on the test split
```
 $ python main.py --mode test
 
```

* Model predictions of an image from the test split
```
 $ python main.py --mode demo --img 7a47e8eacea04e64_52382.jpg
 
```
* Image file names from the test split are available in the comma separated value file named split_test.csv with their corresponding annotations.  

# Acknowledgements

We used a significant amount of code from the retinanet pytorch implementation of https://github.com/yhenon/pytorch-retinanet
For the balanced focal loss implementation, we got inspiration from https://github.com/vandit15/Class-balanced-loss-pytorch
e
