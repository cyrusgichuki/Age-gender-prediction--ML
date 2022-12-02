from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(Y_true, ynew)
#y_test1 = np_utils.to_categorical(y_test, 3)



def plot_confusion_matrix(cm, classes ,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       # print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],fmt),

                 #plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
   # cm_plots=['0','1','2']
  #  plot_confusion_matrix(cm=cm,classes =cm_plots ,title='Confusion Matrix')
print('Confusion matrix, without normalization')
print(cnf_matrix )

####################################
plt.figure()
plot_confusion_matrix(cnf_matrix[0:3, 0:3], classes=[0,1,2],
                      title='Confusion matrix, without normalization')



print("Confusion matrix:\n%s" % confusion_matrix(np.array(Y_true ), ynew))
print(classification_report(np.array(Y_true ), ynew))
