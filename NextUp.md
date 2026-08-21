
1. Read about ROC and AUC curves and include them in the analysis.

2. Add more data augmentation techniques. Rotation, translation, zoom, brightness adjustment, Gaussian noise.

3. Show confusion matrix. (save to database and show in web interface)

4. Include comparison of macro precision between models.

5. Describe the architecture of the model in detail. Create a diagram of the model architecture.

6. Save predictions per class (Validate method) + true label into LCPredictions, for both C# and Python. Needed for ROC/AUC and confusion matrix in web app.

7. Reconsider LCEpochData - feels bloated/oddly shaped (loss + precision + everything crammed together). Maybe split into two classes: one for training classification report, one for validation classification report. Validation one could hold LCPredictions. Just an idea, think it through before touching anything.



