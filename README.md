## Classification of Pneumonia in X-ray/CT Image Reports

**Introduction**
---
- Developed a classification model to support medical professionals in making data-driven decisions
- Aimed to classify radiologic reports into 3 classes: Negative(indicating no pneumonia), Positive, Obscure

**Data**
---
- Consisted of radiology reports of surgical patients from January 2008 to January 2018 in the Asan Medical Center in South Korea
- Included bilingual texts, combining both Korean and English
    <img width="683" alt="Example of radiology reports with bilingual texts" src="https://github.com/eunbyul616/PneumoniaClassification/assets/52561281/039da86d-0843-4d04-a4de-0cec48ab9dc0">

    *Figure 1. Example of radiology reports with bilingual texts*
        
**Objectives**
---
- Developed a robust classification model for pneumonia in radiologic reports
- Addressed challenges related to bilingual texts and enhance the interpretability of the model’s outputs

**Process**
---
1. Preprocessing
    - Performed removal of punctuation, tokenization, and lemmatization
    - Converted symbols to word representing their specific meanings (e.g. ‘→’, ‘R/O’ into ‘therefore’)
      ![Example of radiology reports that included symbols](https://github.com/eunbyul616/PneumoniaClassification/assets/52561281/0d0879ac-292c-44a4-8cc5-1749d1fdcd8e)

      *Figure 2. Example of radiology reports that included symbols*
      
2. Transforming source words to target Words
  - To handle bilingual issues and prevent information loss, source words(Korean) were transformed into target words(English) using two embedding models
  - Employed Singular Value Decomposition(SVD) to align these models into the same space, creating a cross-lingual representation for the classification model
  
3. Developing classification model
  - Developed Bi-LSTM model to capture contextual information
  - Added Attention layer to visualize the attention weight of each word
    
    <img width="672" alt="Visualization of the importance of words by attention weights" src="https://github.com/eunbyul616/PneumoniaClassification/assets/52561281/b4007445-17da-4a3d-9f29-2ed07b634f68">
    
    *Figure 3. Visualization of the importance of words by attention weights*

**Usage**
---
It depends on Tensorflow2, Numpy, scikit-learn, NLTK.
```
python3 train.py --config_file "config.json"
```
