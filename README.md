# opencv-face-gender-age-recognition using Deep Learning
# Models

Download models from

Gender Net : https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0"

Age Net : https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0"

Openface : https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7

# Run Code

### Step#0 - Run face_capture.py to store your face image inside dataset. Maximum run for 2 seconds and press q to quit.

`python face_capture.py --name your_first_name`

### Step#1 - A deep learning feature extractor to generate a 128-D vector describing a face.

`python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7`

### Step#2 - The Linear SVM model will be trained by this script in Step #2. We’ll detect faces, extract embeddings, and fit our SVM model to the embeddings data.

`python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle`

### Step#3 - Face recognize using those trained models alongside gender and age detect/predict

`python face_gender_age_recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle`

# Contact & Support

### email: dsubham776@gmail.com
Never forget to give your opinion, review and suggestion. Have a nice day!
