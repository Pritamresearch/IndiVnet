# IndiVnet

## Overview
We propose IndiVnet, a novel segmentation model designed explicitly for Indian driving scenarios. As Autonomous vehicle technology has witnessed significant advancements globally, challenges unique to India's diverse driving conditions remain unaddressed. Existing segmentation models, tailored for Western environments, fall short of capturing the intricacies of Indian roadways characterized by complexity, congestion, and cultural nuances. To fill this gap, we proposed our model. Evaluation against state-of-the-art models showcases IndiVnet's superior performance, with a Mean Intersection over Union (MIoU) of 0.6998. This research underlines the importance of tailored approaches for autonomous navigation in diverse global contexts, emphasizing the need for region-specific datasets and models to ensure safe and efficient autonomous driving experiences.

## Result
Across the different models, the performance varies for each semantic class. For instance, in the "roadside object" class, DRN ResNet 18 demonstrated the highest TP value of 1,794, while for the "far object" class, IndiVNet exhibited the highest TP value of 10,420. Moreover, the "sky" class saw VGG16 UNet achieving the highest TP value of 9,889.

## Folder Structure
**data_load_and_preprocessing/**: Contains scripts for loading and preprocessing the dataset.
**model/**: Includes scripts for defining the machine learning models. 
**train/**: Contains scripts for training the models.
**test/**: Includes scripts for testing the models.
**IndiVnet main/**:Contains code for end-to-end model implementation using all the scripts.
