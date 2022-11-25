# AML_Project_Perez_Rodriguez

![resultados_cualitativo](https://github.com/IBIO4490/Kvasir_Final_Project-perez-rodriguez/blob/master/images/resultados_cualitativo.jpg)

Robust Mask2Former model on Endovis2018


# Results on the val split

* Quantitatively, our model achieves the following metrics on Endovis2018:

```
mIoU	IoU	mcIoU	                     Instrument Categories						
				BF	PF	LND	VS/SI	GR/CA	MCS	UP
82.96	78.69	48.07	85.01	37.53	46.14	64.23	0.00	93.47	10.10



# Environment and installation
1. Clone this repository 
2. Create a new environment
```
 $ conda activate /media/SSD0/aperezr20/anaconda3/envs/mask2former

# Usage

* Evaluation of the model on the val split
```
 $ python main.py --mode test
 
```

* Model predictions of an image from the test split
```
 $ python main.py --mode demo --img img_file_name
 
```

