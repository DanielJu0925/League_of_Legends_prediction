<h1>League_of_Legends_prediction</h1>

by Bofu Zou and Zhuoxuan Ju

## Framing the Problem

### Problem Identification

Exploring about the prediction on the winning or losing result of a team in one game prior to the result being determined is our focusing point for this project. We would like to form a model that input the relavent data and return the result of that game. We are building a classifier with a binary classification. 

The reason why we choose the topic is because the winning trend in early and mid games are signifcant and can influence the final result of a game. Our model will predict the result of a team in one game by analyzing the team's resources gained(gold, experience(xp), kills(creat a gap of gold earned and xp learned between you and opposite side) and the difference between this team and its opposite team in early game(10 mins) and mid game(15 mins). Our response variable is the result of that game. It will be a binary result which 1 equals to win and 0 equals to lose. The result of each game is the hardest to predict just by watching the game untill the game actually ends (one team destorys the opposing Nexus) and thus we choose it as our response variable.

The metric we take is accuracy. Firstly, the observation doesn't contain class imbalance, since we want to predict one team's result in the game, and it is a binary result. In other words, the winning games and the losing games are equal. The errors (false positive and false negative) are also equally bad (given the confusion matrix below, generated by the baseline model on the testing data). Thus, we choose accuracy over F1-score as to evaluate our model.

The features we would use in our model would be columns generated during the game like `"golddiffat15"` (gold difference at 15 mins). Columns like `"totalcs"` might contain more decisive data, but we can't use it since we are tring to predict the team's result in a game prior to game over while this column is only able to be generated after the game.

In order to explore our prediction problem, we performed the same data cleaning steps as we do in project3: Investigation-on-League-of-Legends. In addition, we drop the rows that are not contains our target values. For example, in the game from the league LPL, the data like `"golddiffat15"` did not being recoreded inside the dataframe, and we drop it in order to not confuse for our model. At last, we drop the irrelevant columns with missing values from the dataframe


## Baseline Model

For our Baseline Model, we used `Pipeline` to encode several features and applied `DecisionTreeClassifier`(set the max_depth to 12). To evaluate our model’s ability to generalize to unseen data, we used `train_test_split()` seperating traning data and testing data. 

We used `"side"`, `"league"`, `"killsat10"`, `"xpdiffat10"`  as the features in our baseline model.

`"side"` - this is one nominal column. We implement one hot encode that particular column in order to get it to fit in our model.

`"league"` - this is a nominal column, and we one hot encoded this column to 1 and 0 for each league in the game.

`"killsat10"` - this is a quantitative column, and it represents the numbers of kills for the team at 10 minute. There will be no change on this column.

`"xpdiffat10"` - this is a quantitative column. It is the difference in experience between two teams at 10 minute.

We fitted the model to the training data that we splitted previously and tested the model on both the training set and testing set. The average score of our trained model is around 0.79 on training data and around 0.58 on testing data.

We believe the model is not good one. First, the accuracy score of testing data is relatively lower than that of the training data, which indicates that the model has overfitting issue(ability to generalize to unseen data is restricted). Second, we solely included four features, which will limit the model's ability to generate more accurate prediction. Last but not least, the features we selected are not highly correlated with the games' results. For instance, we only include the kills and experience at 10 minute, but the first 10 minute cannot fully represent the game, since the game might last much longer than only 10 minute.

There is the confusion matrix for our testing data result:

<iframe src="output.png" width=800 height=600 frameBorder=0></iframe>

## Final Model




## Fairness Analysis

