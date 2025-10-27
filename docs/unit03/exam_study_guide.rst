Exam Study Guide 
================

The exam will cover Unit 1, Unit 2 and Unit 3 up to and including the basics of 
Artificial Neural Networks (ANNs) and dense (Fully-connected) networks. It will 
not include CNNs or any other material after. 

There are two sections to the guide: a general "exam preparation" section, and a 
"sample exam questions" section. The preparation section is intended to be comprehensive 
and give you a guide for the kinds of topics you should understand. The example 
questions section is intended to give you a flavor of the kinds of questions 
you are likely to see on the exam. Note that the examples do not constitute 
a comprehensive set of questions. 


General Exam Preparation 
------------------------

Unit 1
^^^^^^
* Explain the difference between a Pandas Series and a DataFrame. To load data 
  from a CSV file, which would you use? What does the Pandas Dataframe correspond 
  to with respect to the CSV file? What does the Series correspond to? 
* Describe some of the sources for outliers in data. In which cases should outliers 
  be discarded from the dataset, and in which cases should they be included? 
* When do you need to use imputation on a dataset? What is imputation and 
  what is the difference between univariate and multivariate imputation?
* Describe some examples of univariate imputation and some examples of multivariate 
  imputation. What ``numpy`` functions might be used in an implementation? 
* What does the ``groupby`` function do in Pandas? How would you use ``groupby`` 
  to implement a univariate imputation? 

Unit 2
^^^^^^
* Describe the difference between supervised and unsupervised learning. Under which 
  scenarios would you use one versus the other? What are the main advantages and 
  disadvantages of each? Given specific scenarios and datasets, make sure you can 
  identify which should be used.
* What is the difference between classification and regression? Given specific 
  scenarios and datasets, make sure you can identify which should be used.
* How would you identify a linearly separable dataset from a pictorial example? 
  If a dataset is linearly separable, what guarantees could you make? 
* How is a decision function used in a machine learning model? Is it used in classification, 
  regression or both? 
* What is the difference between accuracy, recall, precision, and F-1? Make sure you can 
  identify which metric is most important for a given scenario and be able to justify 
  your answer.  in each of the following examples? Given specific scenarios and datasets, make sure you can identify which should be used and be able to explain your answer. 
* Describe the two primary methods we have discussed in class to optimize different 
  classification metrics. Be sure you are able to answer questions about how to 
* What is the meaning of the “k” in the K-nearest neighbor algorithm? What is a hyperparameter in a machine learning model? What standard method do we use to determine the optimal values of hyperparameters?
* Describe the advantages and disadvantages of K-nearest neighbor versus Linear 
  Classification.
* As the value of “k” in the K-nearest neighbor algorithm increases, how does the model’s 
  sensitivity to outliers change? 
* What is the definition of a hypyerparameter? Describe some example of hypyerparameters in the 
  models we have studied, and know about the ways in which their values can impact the model's 
  performance. 
* Describe cross validation and what it is used for. 
* Describe advantages and disadvantages of the Decision Tree algorithm. How does it 
  compare with Linear Classification? KNN? 
* What is the relationship between Decision Trees and Random Forests? What is the 
  advantage of Random Forests compared to Decision Trees? What is the disadvantage? 
* To what extent is Random Forest an example of an ensemble method? What are other examples 
  of ensemble methods? 

Unit 3 
^^^^^^
* What is the mathematical definition of a perceptron? How do the weights, biases, 
  and activation function factor into the definition?
* What is the role of the activation function in an ANN? 
* What are the advantages of ANNs over the methods we have discussed in Unit 2? 
  What are some of the disadvantages? 
* For a fully connected ANN, how do the input dimensions of layer depends on the 
  output dimension of other layers? 
* How does the input dataset put constraints on the architecture of the ANN? 
  Which layer(s) are constrained? And which parts of the layer(s)? 
* For a classification problem, what are the constraints on dimension on an ANN?

  

Example Exam Questions 
----------------------

.. warning:: 

  This set of example questions is **not** intended to be a comprehensive study guide. Rather,
  it is only intended to give you a sense of the format of questions you will be asked. 
  Be sure to review all of the topics in the previous section. 


Short Answer 
^^^^^^^^^^^^

1. For each of the following scenarios, specify whether the problem would best be solved 
   as a supervised or unsupervised learning problem.
  * An e-commerce company wants to build a model to segment its customers into groups with 
    similar purchasing patterns, without any predefined categories.
  * A hospital has a dataset of patient blood test results, and each record indicates whether or not 
    the patient was later diagnosed with diabetes. They want to train a model to predict 
    if a new patient has diabetes based on their blood test results.
  * A speech recognition team wants to build a model that converts spoken audio clips 
    into text, using a dataset of audio clips paired with their corresponding 
    transcriptions.
  * An online streaming music company has the listening history of each of its users. It 
    would like to build a model to identify groups of users with similar listening habits. 

2. True/False
  * A Linear Regression model makes use of decision functions and the perceptron algorithm. 
  * A dataset contains information about the lengths of flower pedals, but some of the values 
    are missing. Replacing the missing flower pedal lengths with the median of all lengths is 
    an example of univariate imputation. 
  * Decision Trees are an example of an ensemble method in machine learning. 
  * A medical lab is training a machine learning model to predict whether a patient will be 
    eligible for a new treatment. The treatment is cheap and very safe. The lab should 
    evaluate and select the best model using the recall metric to minimize false negatives.

3. Multiple choice 
  * The K-nearest Neighbor algorithm:

    a) Is one of the most accurate machine learning models
    b) Can learn non-linear decision boundaries
    c) Does not have any hyperparameters, so it is fast to train
    d) Works best with recall
    e) None of the above 

  * The F-1 metric minimizes: 

    a) False positives 
    b) False negatives 
    c) Precision 
    d) Recall 
    e) None of the above  

  * When defining a ``Dense`` layer object in a ``Sequential`` Artificial 
    Neural Network (ANN) using the Keras API, one must always:

    a) Pass the input dimension as an argument
    b) Ensure the layer has more perceptrons than its input dimension
    c) Specify the batch size to use
    d) Specify the number of perceptrons in the layer
    e) None of the above 

Longer Answer 
^^^^^^^^^^^^^

1. What is the purpose of cross validation in machine learning?  

2. In the context of machine learning explain the difference between 
   training accuracy, validation accuracy and test accuracy.

3. Explain the difference between the Decision Tree algorithm and the Random Forrest 
   algorithm. What are the strengths and weaknesses of each? 

4. When would you use ``RobustScaler`` and when would you use ``StandardScalar`` on 
   a data set? 
   

Code Analysis 
^^^^^^^^^^^^^

1. What will be the output of the following code? 

.. code-block:: python3 

  df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [10, 20, 30]
  })

  df["A"] = df["A"].map(lambda x: x * 2)
  print(df)

2. What is the output of the following code?

.. code-block:: python3 

  cars = pd.DataFrame({
  "brand": ["Toyota", "Toyota", "Tesla", "Tesla"],
  "price": [20000, 25000, 80000, 90000]
  })
  print(cars.groupby("brand")["price"].apply(sum))

3. Your friend tells you they just figured out a clever way to 
   improve the accuracy on their class project that uses K-nearest
   Neighbor. They show you their main loop:

.. code-block:: python3 

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
      
      best_model = none 
      best_k = 0
      best_accuracy = 0

      for k in np.arange(1, 100):
        m = knn = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, m.predict(X_test))
        if accuracy > best_accuracy:
          best_model = m 
          best_k = k 
          best_accuracy = accuracy 
      print(f"The best model is: {best_model}")  
  
The syntax is correct and the code produces a best model, but what is the 
flaw in your friend's approach? 

Code Authoring 
^^^^^^^^^^^^^^

1. You want to develop an Artificial Neural Network to classify images as containing 
   cats, dogs or neither. You have a set of labeled images that grey scale (i.e., one 
   channel of intensity with value between 0 and 255) of size 1000x728 pixels.

   Your network should be a dense (i.e., full-connected) network with three layers: 
   one input layer, one hidden layer, and one output layer. Write the code to construct 
   such an ANN using the Keras API. In this section, you do not need to train your model, 
   only define the archtiecture. You can use the following code snippets, but note, you 
   may not need or want to use all of them. 

   .. code-block:: python3

    Dense(?, input_dimension=(?), activation=?)
    
    from tensorflow.keras import Sequential

    m = Sequential()

    image_size = 1000*728

    image_dimension = 1000*728*255

    from tensorflow.keras.layers import Dense

    "tanh", "relu", "softmax", optimizer="adam", loss='categorical_crossentropy'
    
    from tensorflow.keras.utils import to_categorical

    m.add(?)

2. You want to build a model to detect whether AI was used on an exam, which is not 
   allowed and constitutes academic dishonesty (cheating).

   Write code to perform a hyperparameter search for the optimal Random Forest classifier 
   model. Your search should use explore a space that includes random forests containing 
   anywhere from 2 to 100 trees, that have a maximum depth between 2 and 10 levels and 
   that consider leaves with a minimum set of samples between 2 and 5.   

   What is the best metric to use for this use case? Explain your answer and optimize 
   your grid search for this metric. 

   You may want to use some of (but not all of) the following code snippets in your solution: 

  .. code-block:: python3

    gscv = GridSearchCV(model, param_grid, cv=?, n_jobs=4, scoring=?)

    from sklearn.tree import DecisionTreeClassifier

    param_grid = { . . . } 
    
    "min_samples_leaf": np.arange(start=?, stop=?)

    gscv.fit(X_train, y_train)
    
    from sklearn.ensemble import RandomForestClassifier

    model = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
    
    model = RandomForestClassifier(random_state=?)
    
    from sklearn.model_selection import GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
    "n_estimators": np.arange(start=?, stop=?, step=?)
    
    "max_depth": np.arange(start=2, stop=20),

  



