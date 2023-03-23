<h1>League_of_Legends_prediction</h1>

by Bofu Zou and Zhuoxuan Ju

## Framing the Problem

### Problem Identification

Exploring about the prediction on the winning or losing result of the game is our focusing point for this project. We would like to form a model that input the relavent data and return the result of that game. We are building a classifier with a binary classification. 

Our response variable is the result of that game. It will be a binary result which 1 equals to win and 0 equals to lose. The reason why we choose it is because inside League of Legends, no matter what resources they receive in the game, the result of the game should be the most important thing for either players and audiences. In addition, the result of each game is the hardest to predict just by watching the game untill the game actually ends (one team destorys the opposing Nexus), and thus we choose it as our response variable. All the columns' data are provided and collected before the game ends which are the responsible features we can include in our model. 

The metric we take is accuracy. Firstly, the observation doesn't contain class imbalance, since we want to predict one team's result in the game, and it is a binary result. In other words, the winning games and the losing games are equal. So the errors (false positive and false negative) will not cause a huge influence on our result, and there is no class imbalance inside our data. Both false positive and false negative are equally bad to us.

In order to explore our prediction problem, we performed the same data cleaning steps as we do in project3: Investigation-on-League-of-Legends. We still keeping all the team rows inside the dataframe, and we strip all the rows for specific player, but we also include some new data cleaning.


## Baseline Model

For our Baseline Model, we used `Pipeline` to encode several features and applied `DecisionTreeClassifier`(set the max_depth to 12). To evaluate our model’s ability to generalize to unseen data, we used `train_test_split()` seperating traning data and testing data. 

We used `"side"`, `"league"`, `"killsat10"`, `"xpdiffat10"`  as the features in our baseline model.

`"side"` - this is one nominal column. We implement one hot encode that particular column in order to get it to fit in our model.

`"league"` - this is a nominal column, and we one hot encoded this column to 1 and 0 for each league in the game.

`"killsat10"` - this is a quantitative column, and it represents the numbers of kills for the team at 10 minute. There will be no change on this column.

`"xpdiffat10"` - this is a quantitative column. It is the difference in experience between two teams at 10 minute.

We fitted the model to the training data that we splitted previously and tested the model on both the training set and testing set. The average score of our trained model is around 0.79 on training data and around 0.58 on testing data.

We believe the model is not good one. First, the accuracy score of testing data is relatively lower than that of the training data, which indicates that the model has overfitting issue(ability to generalize to unseen data is restricted). Second, we solely included four features, which will limit the model's ability to generate more accurate prediction. Last but not least, the features we selected are not highly correlated with the games' results. For instance, we only include the kills and experience at 10 minute, but the first 10 minute cannot fully represent the game, since the game might last much longer than only 10 minute.


## Final Model



## Fairness Analysis

