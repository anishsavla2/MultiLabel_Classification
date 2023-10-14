## Multi-label Text Classification for Movie-related Data

### Problem Overview:

The objective here is to tackle the issue of text classification, specifically focusing on multi-label data. For strings with multiple labels, the goal is to determine all relevant labels and order them alphabetically in the predictions. The provided dataset encompasses utterances with movie-related information. From the list of strings, the task is to identify tags associated with movie names, actor names, producer names, director names, movie countries, movie locations, movie budgets, actor genders, and more. The dataset output is pre-determined tags.

### Model Preparation:

**Features:** The primary features utilized for model training include:
- Filtered user utterances.
- Processed core relations.

**Processing Steps:** Original utterances underwent a series of processing steps:
1. Duplicate utterances were removed.
2. Stop words were eliminated.
3. Backslash-apostrophes were removed.
4. All characters, except alphabets, were removed.
5. White spaces were purged.
6. The entire text was converted to lowercase.

After filtering the utterance text, TF-IDF vectorization was employed.

**Tag Processing:** 
- Initially, multiple tags were split to be treated as distinct entities.
- Some tags underwent format changes (e.g., "producedby" was changed to "produced.by").
- The MultiLabelBinarizer was used on tags, converting them into binary values (0 or 1). The assignment of these values depended on the utterance.

### Experiments:

The training data underwent a train-test split using `sklearn.model_selection`, generating both training and evaluation datasets. The previously applied TF-IDF vectorization was used to weigh each word in the utterance.

Training and evaluation data (from the split) were fed into the model. Predictions were made on both the test data (from the split) and the actual test set sourced from "testdata.csv". The `sklearn.metrics` module from Scikit-learn was employed to measure F1 scores and accuracy on the training data. Model performance (using the F1 score) influenced the competition submissions on Kaggle.

### Results:

From the training data split:
- The F1 score for the Random Forest model stood at 0.70.
- For the Decision Tree, the F1 score was 0.67.

However, there was a decline in the Random Forest's performance, with the F1 score dropping from 0.70 to 0.63 when assessed against the test data.
