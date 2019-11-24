# DigitalFace Makeup
    This project applies the make up from a reference image to a target on a pixel by pixel basis. The reference and target
    were aligned by warping using 77 detected face landmarks. The images were decomposed into large scale layer, detail layer 
    and color layer through which makeup highlights and color information were transferred by Poisson editing, weighted means and alpha blending. 
    The test results showed that the techniques work wellwith reference images of a different age, a different skin color and even a hand-sketched reference image.
    
     ![alt text](./SampleImages/Example)

### Prerequisites

      * dlib==19.18.0
      * numpy==1.17.4
      * opencv-python==4.1.2.30
      * scipy==1.3.3

  Install the above requirements by 'sudo pip3 install requirements.txt'


```
Give examples
```

## Installing

      * Clone this repo in your local machine usine 'git clone https://github.com/akki111singh/Make_Up.git'
      * Install all the requirements using 'pip3 install requirements.txt'

## Running the tests
      * Do 'python3 Makeup.py source refrence'
      * Select the Forehead points of Source Image using mouse
      * Similarly select the Forehead points of thre refrence image
      * After some time the Final output will be displayed'
      * Press 'shift key' to exit the program'


## Contributing
      
      * Fork the project.
      * Clone this repository to your local machine.
      * Now add upstream by using command - **git remote add upstream "name of my repo"**
      * Create a new branch on your local machine.
      * Start contributing and make a pull request to apply these changes.
   

See also the list of [contributors](https://github.com/Make_Up/contributors) who participated in this project.


## Refrences
      * [DigitalFace Makeup by Example](https://www.comp.nus.edu.sg/~tsim/documents/face_makeup_cvpr09_lowres.pdf) Dong Guo and Terence Sim 

