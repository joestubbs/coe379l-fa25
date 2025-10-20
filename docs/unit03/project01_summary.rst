Project 1 Summary 
=================

Overall the class did an excellent job! 

26/34 were 19 or higher!

* 20 (16 projects)
* 19+ (10 projects): 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19, 19, 19
* 18+ (2 projects): 18.25, 18
* 17+ (4 projects): 17.75, 17.5, 17, 17
* 16+ (1 projects): 16
* 15+ (1 projects): 15


General Comments: Kudos and Cautions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Overall, the class performed very well on Project 1 — great work! Many of you demonstrated 
a strong understanding of data exploration, feature analysis, and model development.  The effort put into 
cleaning the data, visualizing patterns, and explaining results was clearly reflected in your scores.

Kudos
^^^^^^
1. Several of you provided excellent insight into the dataset, especially observations about how the age of the animals 
   relates to their outcome type. This is a very insightful and eye-catching finding. Such abservations help identify 
   relevant features that should be considered when building the mode.

2. Many reports had well-structured sections with clear explanations supported by data and logical reasoning. The organization 
   of your Jupyter Notebooks was very clear, and the written reports were easy to follow with strong supporting statements.

Cautions 
^^^^^^^^
1. Many students did not drop the Outcome Subtype column when splitting or training the data. Keep in mind that Outcome Subtype 
   is essentially a more detailed version of the Outcome Type (the label). If you include it as a feature, you are indirectly giving 
   the model the correct answer, which results in data leakage and makes the model’s performance unrealistic.

2. Some notebook cells could not be executed, for example, univariate plots did not display or models take very long time to finish training. 
   This is due to overly high-dimensional feature sets. The project is intentionally structured step-by-step: clean and preprocess the data, 
   exploring and find patterns, and then build and train the model. 

3. When using AI tools to assist with coding or explanations, make sure you understand what the generated code is doing. 
   Do not rely on AI outputs blindly — use AI as a tool, not a replacement for understanding.

4. Please remember to include the Use of AI document in future projects — even if you did not use AI. In that case, 
   simply state that AI was not used. This is required for record-keeping purposes.
