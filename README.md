# Pakistani NumberPlate Detection, Tracking and Recognition

###  Features
- YOLOv5 Object Tracking Using Sort Tracker
- detection in specific box
- ANPR with OCR
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported


### Environment
- Create a Anaconda Envirnoment 
```
conda create -n anpr python=3.10
conda activate anpr

pip install PyYAML scipy pandas matplotlib tqdm seaborn
pip install filterpy
```
If you have GPU use this:
    
```

"To install pytorch with cuda make sure you had already installed and set the path cuda & cudnn in base of your system."

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
If you have CPU use this: 
```
pip3 install torch torchvision torchaudio
```

```
pip install easyocr
pip install opencv-python
```

For Inference use this


```
python main_rect_det.py --weights our.pt --source test.mp4 --view-img 
```
For tracking code will be uploaded soon

