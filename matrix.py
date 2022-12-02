# imports
import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


y_true = ["12", "ball", "ball", "bat", "bat", "bat"]
y_pred = ["bat", "bat", "ball", "ball", "bat", "bat"]
conf_matrix = (confusion_matrix(y_true, y_pred, labels=["bat", "ball"]))

# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='Greens_r')


# labels the title and x, y axis of plot
fx.set_title('gender age Confusion Matrix using Seaborn\n\n');
fx.set_xlabel('Predicted gender age')
fx.set_ylabel('Actual gender age ');

# labels the boxes
fx.xaxis.set_ticklabels(['False','True'])
fx.yaxis.set_ticklabels(['False','True'])

plt .show()