<h1>League_of_Legends_prediction</h1>

by Bofu Zou and Zhuoxuan Ju

## Framing the Problem

### Problem Identification

Exploring about the prediction on the winning or losing result of the game is our focusing point for this project. We would like to form a model that input the relavent data and return the result of that game. We are building a classifier with a binary classification. 

Our response variable is the result of that game. It will be a binary result which 1 equals to win and 0 equals to lose. The reason why we choose it is because inside League of Legends, no matter what resources they receive in the game, the result of the game should be the most important thing for either players and audiences. In addition, the result of each game is the hardest to predict just by watching the game untill the game actually ends (one team destorys the opposing Nexus), and thus we choose it as our response variable. All the columns' data are provided and collected before the game ends which are the responsible features we can include in our model. 

The metric we take is accuracy. Firstly, the observation doesn't contain class imbalance, since we want to predict one team's result in the game, and it is a binary result. In other words, the winning games and the losing games are equal. So the errors (false positive and false negative) will not cause a huge influence on our result, and it won't have some severe consequences.

In order to explore our prediction problem, we performed the same data cleaning steps as we do in project3: Investigation-on-League-of-Legends. We still keeping all the team rows inside the dataframe, and we strip all the rows for specific player, but we also include some new data cleaning.

## Baseline Model

For our Baseline Model, we used `Pipeline` to encode several features and applied `DecisionTreeClassifier`(set the max_depth to 12). To evaluate our model’s ability to generalize to unseen data, we used `train_test_split()` seperating traning data and testing data. 

We used `"side"`, `"league"`, `"dragons"`, `"visionscore"`  as the features in our baseline model.

`"side"` - this is one nominal column. We implement one hot encode that particular column in order to get it to fit in our model.
`"league"` - this is a nominal column, and we one hot encoded this column to 1 and 0 for each league in the game.
`"dragons"` - this is a quantitative column, and it represents the numbers of dragons that each team kill. There will be no change on this column.
`"visionscore"` - this is a quantitative column. It is the score each team on maintaining visions, and we cast no transformation on it.

The average score and result of the performance of our trained model on testing data is around 74%, and we believed that although the score seems to be fine, but the model is not good enough, because we solely included four features, but numbers of other features can influence the result of the game in each match. We cannot predict the correct result everytime with limited information. 

## Final Model

## Fairness Analysis

