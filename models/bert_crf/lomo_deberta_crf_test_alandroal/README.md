# Vote NER Model

This model was trained for Portuguese vote identification Named Entity Recognition.

## Model Details
- Architecture: BertCRFNER
- Number of labels: 17
- Base model: neuralmind/bert-large-portuguese-cased

## Labels
- O
- B-COUNTING-MAJORITY
- B-COUNTING-UNANIMITY
- B-SUBJECT
- B-VOTER-ABSENT
- B-VOTER-ABSTENTION
- B-VOTER-AGAINST
- B-VOTER-FAVOR
- B-VOTING
- I-COUNTING-MAJORITY
- I-COUNTING-UNANIMITY
- I-SUBJECT
- I-VOTER-ABSENT
- I-VOTER-ABSTENTION
- I-VOTER-AGAINST
- I-VOTER-FAVOR
- I-VOTING

## Usage
Load this model using the Vote NER framework or convert to standard HuggingFace format.

## Training Information
- Best F1 Score: 0.9065
- Training completed at step: 2650
