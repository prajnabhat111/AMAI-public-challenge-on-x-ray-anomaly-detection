# AMIA-public-challenge-on-x-ray-anomaly-detection
X-ray anomaly detection https://www.kaggle.com/competitions/amia-public-challenge-2024/overview

The problem at hand is object detection and classification. Images in the dataset
 can contain one or more thoracic abnormalities or nothing at all. We are required
 to identify the abnormalities, if exist, and indicate the coordinates of the box
 bounding the area where it’s seen.

The dataset consists of a total of 15,000 scans that have been annotated by expe
rienced radiologists. From those, 8,573 images have been used for training and
 6,427 images for testing. The corresponding CSV file with metadata for train
ing images contains image filenames, class names, radiologist IDs, and bounding
 box coordinates. The CSV file for test images contains image filenames.

  This problem can be seen from 3 following perspectives.
 1. Simple Multi-label Classification Problem: This task involves a sim
ple classification of images into 14 classes of abnormalities or if no abnor
malities exist.
 2.  Object Localization and Classification: This involves classifying the
 images into different classes and drawing bounding boxes around the area
 of abnormalities.
 3.  Object Localization and Classification considering Radiologists
 data: This task is a complicated task where the object localization and
 classification should take into consideration, the same images belonging
 to the same classes, but annotated by different radiologists resulting in
 different bounding box coordinates.
 
 To keep the analysis and implementation rather simple, I have implemented
 the first 2 tasks separately in notebooks - and -.

### Simple Multi-label Classification Problem

  First, all irrelevant columns were dropped, leaving only columns with image IDs
 and their label information. Since each image could have belonged to several
 classes, the data was grouped by the former, and one-hot encodings were generated for the labels of each image. After preprocessing data in this manner, the Densenet101 was chosen for training but without pre-trained weights. This
 was appropriate since it let the model learn from scratch features that are specific to the new dataset, thus avoiding biases of the dataset originally used in
 pertaining. Afterward, a deep architecture from Densenet101 was trained on
 newly prepared data to effectively capture complex patterns and relationships
 in the images. Since the problem on Kaggle contained test data without the
 label information, I have split the existing training data into 2 sets with the
 idea of checking model accuracies.

#### Results 
After training the model for 10 epochs, the training accuracy achieved was
 92.15% and loss incurred was 0.1672 using the BCEWithLogitsLoss loss func
tion. The test accuracy was also seen to be 91.04% indicating the model was
 performing decently well. The threshold on top of the sigmoid function to get
 the prediction probabilities was set to 0.5.
 
 The figure shows the loss and accuracy curves plotted against epoch runs.

 
  ![Classification Result](https://github.com/prajnabhat111/AMAI-public-challenge-on-x-ray-anomaly-detection/blob/main/Images/Result%20Classification.png?raw=True "Classification Result")

 #### Future Enhancements 

 Although this task already seems to return quite good accuracy, there is always
 scope for betterment. Since we deal with medical data here, excellent accuracy
 in practice is indispensable.
 Several strategies can be looked upon in case of potential betterment for
 improving the accuracy of the model. One is to apply data augmentations to
 avoid ”Clever Hans” behavior—a model learning to exploit spurious patterns
 in the data and not genuinely understanding real features of the data. Another
 would be to ensure a balanced dataset, for it is capable of preventing bias toward
 overrepresented classes by the model. The threshold can be set to varying
 values based on some strategy to avoid any biases due to class imbalance. Also,
 different architectures that could turn out to be simpler or more complex can be
 tried, providing the model with an opportunity to fit intricate patterns better
 in the images. Finally, if domain-specific features were included with the task,
 such as those found in medical imaging, a model pre-trained on medical data
 rather than typical RGB images would provide a more relevant starting point.

 ### Object Localization and Classification

 In handling this object localization and classification problem, all radiologist IDs
 were removed from the dataset for simplification. 
 I used a pre-trained model of Faster R-CNN with a ResNet-50 backbone.
 The choice will be well conditioned for the problem because it has robust performance in the identification of objects within the image and correspondingly
 accurately classifying them. Faster R-CNN architecture is designed to become
 outstanding in tasks performing accurate object detection; therefore, harnessing
 the deep feature extraction abilities of ResNet-50 to capture detailed features
 of an image while efficiently proposing probable object locations by the Region
 Proposal Network. This thus makes this combination very ideal for scenarios
 where accurate bounding box predictions and class labels matter.

 ![Object Detection Result](https://github.com/prajnabhat111/AMAI-public-challenge-on-x-ray-anomaly-detection/blob/main/Images/Result%20Object%20Detection.png?raw=True "Object Detection Result")

  #### Results 

  Due to limited computational resources, I trained this for only 5 epochs, the
 training result gave a loss of 0.2047 and a test accuracy ranging between 40%
 and 50%, however, by looking at classification results only. Certain steps were
 followed after the predictions were made by the model, such as filtering out
 the predictions with a confidence score of less than 40%. This threshold was
 selected after trying out a few other options like using 0.3 and 0.5 for thresholds
 reduced the accuracies. A noteworthy observation made here was again leading
 to an unbalanced dataset where a lot of images were classified as ’no finding’.
 
 I tried reducing the dataset size by balancing the data from all the classes.
 However, this did not show good results.
 
 Figure 3 shows the loss curve plotted for epoch runs.

 ![Visualisation Image](https://github.com/prajnabhat111/AMAI-public-challenge-on-x-ray-anomaly-detection/blob/main/Images/Visualisation%20Image.png?raw=True "Visualisation Image")

 #### Future Enhancements 

 Handling imbalanced data requires more effective strategies than simply reduc
ing the dataset size, as this approach significantly limits the available data and
 can lead to inefficiencies. More sophisticated techniques should be taken into
 consideration. For instance, the utilization of class weights in the loss function
 makes sure that the model pays due attention to the underrepresented classes.
 More than this, threshold values can be further calibrated for each class. That
 ”no findings” is poorly detected suggests the model will benefit from more ad
vanced techniques either in targeted data augmentation or synthetic data pro
duction methods. These methods can lead to a more robust model that will
 be able to handle imbalanced dataset subtleties. Other techniques which may
 improve the model can be implemented as previously suggested. More powerful
 tools related to the localization and classification of an object could be exploited
 and benchmarked.
 
 Finally, the third approach, which is the metadata of radiologists, has not
 been done yet. This should deliver a better model and be able to give a robust
 solution to the actual problem.
 
