## Face Spoofing Detection with Highlight Removal Effect and Distortions
Created by [Inhan Kim](http://imlab.postech.ac.kr/members.htm) and [Daijin Kim](http://imlab.postech.ac.kr/members_d.htm) at [POSTECH IM Lab](http://imlab.postech.ac.kr)

### Overview
With rapid development of face recognition and detection techniques, the face has been frequently used as a biometric to find illegitimate access. It relates to a security issues of system directly, and hence, the face spoofing detection is an important issue. However, correctly classifying spoofing or genuine faces is challenging due to diverse environment conditions such as brightness and color of a face skin. Therefore we propose a novel approach to robustly find the spoofing faces using the highlight removal effect, which is based on the reflection information. Because spoofing face image is recaptured by a camera, it has additional light information. It means that spoofing image could have much more highlighted areas and abnormal reflection information. By extracting these differences, we are able to generate features for robust face spoofing detection. In addition, the spoofing face image and genuine face image have distinct textures because of surface material of medium. The skin and spoofing medium are expected to have different texture, and some genuine image characteristics are distorted such as color distribution. We achieve state-of-the-art performance by concatenating these features. It significantly outperforms especially for the error rate.


### Citation

If you're using this code in a publication, please cite our papers.
```     
Kim, I., Ahn, J., & Kim, D. (2016). Face spoofing detection with highlight removal effect and distortions. 2016 IEEE International Conference on Systems, Man, and Cybernetics (SMC). doi:10.1109/smc.2016.7844907
```


### Overall

   <img src="https://github.com/kimna4/FaceSpoofingDetection/resources/pipeline.PNG?raw=true" width=640>



### Licence

This software is for research purpose only.

Check LICENSE file for details.


