# TACO_MJS
This multi-label classification model is based on the TACO dataset (http://tacodataset.org/). The aim is to detect different kind of trash, this can be used for automatic trash separation. I want to see the impact of image augmentation on model performance. 

This is inspired by the amazing project with Neatleaf, where we could unfortunately not showcase our code on gitHub, as we were working on their code base and with their images. As a result I am thrilled to apply the gained knowledge on a new dataset, a topic that is also close to my heart, preserving and protecting nature.


## :construction: WORK IN PROCESS :construction:


# Workflow

1. Sight the data and the data structure

Instead of the data beeing dispersed in 15 different folders (not sorted by classes - random) and the annotations beeing in a json file, I prefer to have the images all in one folder and the corresponding annotations in a csv file to more easily select labels or split the dataset.

2. Feature engineering & balancing

There are 29 super-categories and 60 categories labelling every kind of trash, e.g. glass bottle, plastic bottle or six pack rings. In my opinion the ultimate goal is to separate trash by material. For this and computational reason I introduced the following 6 categories: plastic, aluminium, paper, glass, other_objects (e.g. shoes) and toxic (batteries).

Some categories appear often like glass bottles (150+) and others are rare. Tackling the problem I partially balanced the data by oversampling (duplicating) rare categories. The file 'train_material_partially_balanced_2023-04-13.csv' is now partially balanced and the rare cases are duplicated so all features have roughly the same (70%) proportion. This increased the row count from 1053 to 4234 rows.

3. Set up base model

Run the first models logging the performance with Mlflow.

4. Apply image augmentation

5. Hyperparameter tuning (e.g. exponential learning late)

6. Use Explainable AI to see what the model is seeing



