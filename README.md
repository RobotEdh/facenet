# facenet
Face detection and regognition using TensorFlow

# references:

 - https://arxiv.org/pdf/1503.03832.pdf 
 - http://www.aisangam.com/blog/real-time-face-recognition-using-facenet/
 - https://github.com/davidsandberg/facenet
  
# Pre-trained models:

Model name	LFW accuracy	Training dataset	Architecture
20180402-114759	0.9965	VGGFace2	Inception ResNet v1

# Commands:


### Transform original images to face images 160x160 to be processed by the network

 ```
 align/align_dataset_mtcnn.py ../data/images  ../data/images_160
```

### Display the 512 values of the vector determined from the face image 160x160

 ```
 face_embeddings_demo_edh.py --img .\IMG_2127.png --modeldir ..//20180402-114759/20180402-114759.pb
```

### Compute the euclidian distance between 2 vectors determined from 2 face images 160x160

  ```
  face_match_demo_edh.py --img1 .\IMG_2127.png  --img2 .\IMG_1896.png --modeldir ..//20180402-114759/20180402-114759.pb
  ```

### Train own dataset using a frozen graph and stroing model and labels in a pickle output

  ```
  classifier.py  TRAIN ../data2/ ../20180402-114759/20180402-114759.pb ..//my_classifier.pkl --batch_size 1000 --image_size 160
  ```

### Identify an image and Display the 512 values of the vector determined from this image

  ```
  identify_face_image_edh.py --img bb.jpg --modeldir ../20180402-114759/20180402-114759.pb --classifier_filename ../my_classifier.pkl
  ```
