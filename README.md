This project is to understand the motivations behind arrival times to work based on a survey of employees conducted in Rennes, France. The data is then used to model the decision-making process of employees regarding their ability to shift their regular arrival times in order to reduce congestion at peak hours. The code 'modeling_benchmarking' looks at ensemble modelling techniques such as Random Forest and Gradient Boosting, as well as the Neural Network Framework. 
Based on metrics such as Accuracy, Recall, Precision and F1 score, the best model is chosen. Learning Curves are also shown.

To run:
Simply run the python file model_benchmarking.py. It automatically calls the related functions to preprocess data and the attention layer of neural network model.

1. The file shift_class_analysis_general analyses the socio economic characterstics of the 'Shift' class in terms of duration and direction of shift.
2. The file shift_class_gender_analysis analyses the genderwise differences in shift duration and direction of the commuters.
3. Finally the file impact_of_shift demonstrates the redstribution of peak congestion to off peak slots when shift class commuters are moved according to their declared preferences in duration and direction

To run any of the files:
Simply run the python file. It automatically calls the related functions to preprocess data. Note that the files require the survey dataset which presently is not made available for public use.
