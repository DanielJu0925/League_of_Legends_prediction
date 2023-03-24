<h1>League of Legends: Prediction of Teams' Result in Tier1 Leagues Matches</h1>

by Bofu Zou and Zhuoxuan Ju

Our exploratory data analysis on this dataset can be found: <a href="https://danielju0925.github.io/Investigation-on-League-of-Legends/">here</a>

## Framing the Problem

### Problem Identification

Exploring about the prediction on the winning or losing result of a team in one game prior to the result being determined is our focusing point for this project. We would like to form a model that input the relavent data and return the result of that game. We are building a classifier with a binary classification. 

The reason why we choose the topic is because the winning trend in early and mid games are signifcant and can influence the final result of a game. Our model will predict the result of a team in one game by analyzing the team's resources gained(gold, experience(xp), kills(creat a gap of gold earned and xp learned between you and opposite side) and the difference between this team and its opposite team in early game(10 mins) and mid game(15 mins). Our response variable is the result of that game. It will be a binary result which 1 equals to win and 0 equals to lose. The result of each game is the hardest to predict just by watching the game untill the game actually ends (one team destorys the opposing Nexus) and thus we choose it as our response variable.

The metric we take is accuracy. Firstly, the observation doesn't contain class imbalance, since we want to predict one team's result in the game, and it is a binary result. In other words, the winning games and the losing games are equal. The errors (false positive and false negative) are also equally bad (given the two confusion matrix plots below, generated by the baseline model and final model on the testing data). Thus, we choose accuracy over F1-score as to evaluate our model.

The features we would use in our model would be columns generated during the game like `"golddiffat15"` (gold difference at 15 mins). Columns like `"totalcs"` might contain more decisive data, but we can't use it since we are tring to predict the team's result in a game prior to game over while this column is only able to be generated after the game.

In order to explore our prediction problem, we performed the same data cleaning steps as we do in project3: Investigation-on-League-of-Legends. In addition, we drop the rows that are not contains our target values. For example, in the game from the league LPL, the data like `"golddiffat15"` did not being recoreded inside the dataframe, and we drop it in order to not confuse for our model. At last, we drop the irrelevant columns with missing values from the dataframe


## Baseline Model

For our Baseline Model, we used `Pipeline` to encode several features and applied `DecisionTreeClassifier`(set the max_depth to 12). To evaluate our model’s ability to generalize to unseen data, we used `train_test_split()` seperating traning data and testing data. 

We used `"side"`, `"league"`, `"killsat10"`, `"xpdiffat10"`  as the features in our baseline model.

`"side"` - this is one nominal column. We one hot encode that particular column in order to get it to fit in our model.

`"league"` - this is a nominal column, and we implement one hot encoding this column to 1 and 0 for each league in the game.

`"killsat10"` - this is a quantitative column, and it represents the numbers of kills for the team at 10 minute. There will be no change on this column.

`"xpdiffat10"` - this is a quantitative column. It is the difference in experience between two teams at 10 minute.

We fitted the model to the training data that we splitted previously and tested the model on both the training set and testing set. The average score of our trained model is around 0.79 on training data and around 0.58 on testing data.

We believe the model is not good one. First, the accuracy score of testing data is relatively lower than that of the training data, which indicates that the model has overfitting issue(ability to generalize to unseen data is restricted). Second, we solely included four features, which will limit the model's ability to generate more accurate prediction. Last but not least, the features we selected are not highly correlated with the games' results. For instance, we only include the kills and experience at 10 minute, but the first 10 minute cannot fully represent the game, since the game might last much longer than 20 or 30 mins.

There is the confusion matrix for our testing data result:

<iframe src="output.png" width=800 height=600 frameBorder=0></iframe>


## Final Model

In our Final Model, we add these new features:

`"killsat15"` - this is one quantitative column that record the kills from the team at 15 minutes from the game start.

`"xpdiffat15"` - this is one quantitative column which represent the experience the team get at the first 15 minutes.

`"opp_killsat15"` - this is a quantitative column, and it is the number of kills from the opposite team at the first 15 minutes.

`"golddiffat15"` - this is a quantitative column, and it is the difference in the gold recevied between two teams at the first 15 minutes.

`"csdiffat15"` - this is a quantitative column that records the difference of the soliders that two teams kill in the first 15 minutes.

`"goldat15"` - this is a quantitative column that the gold the team received in the first 15 minutes.


In order to get a better model, we delete the feature `"league"`, because the different leagues that the games are in is not a variable that will affect our result. For example, the game content will be the same for the match in LPL and LCK. Also, we changed the features `"killsat10"` and `"xpdiffat10"` to `"killsat15"` and `"xpdiffat15"`, because we want to get higher accuracy for the prediction, and we believe the datas that are collected at the first 15 minutes can give our model more information on the game's trend than at 10 minutes, and thus the model will work better with these two new features on predicting the game result.

We added more features. Firstly `"opp_killsat15"`, as the description above, it records the kills from the opposite team at the first 15 minutes. We can compute the kill difference at 15 min with `"killat15"` feature. The kill difference feature can represent the team's combat status and give us a brief conclusion of this team's winning trend aginst the opposite. With that information, we can somewhat increase the complexity of the model without causing overfitting for our model because the match is more complicated than just determined by kills.

Secondly, `"golddiffat15"` represents the difference in the amount of gold received in game for both team in the first 15 minutes. `"goldat15"` shows the gold the team gain. In the game, gold is used for purchasing powerful items to strengthen their power. The higher amount of gold, the stronger the players will be. The economy gap, represented by `"golddiffat15"`, can also indicate if the team has an advantage at 15 mins. So with this information, the model can include more variables for predict the result of the game. 

Lastly `"csdiffat15"`, this feature represents the numbers of soldiers that the team kills in the first 15 minutes. If this features is high, it can represent this team have potentially higher advantages than their opponent, since they have a higher priority in the lane and more gold earned. This will add more chance to win for this team. Thus, we consider it as one of our feature.


For our model, we chose `DecisionTreeClassifier()` as our prediction model, and we will return in a binary result. Inside our model, we split the data into training and testing model and construct the `ColumnTransformer()`. Specifically, we one hot encoded the `"side"`, and using `Binarizer()` with different thresholds (using cross validation to find the most appropriate threshold combos) to binarize both `"golddiffat15"` and `"xpdiffat15"`. Applying `StandardScaler()` on the `"goldat15"` which can standardize the feature. Later on, we form a `FunctionTransformer()` which calculates the kill difference between two teams at 15min. In the next step, we apply one `GridSearchCV()` in order to find the best paramters for `DecisionTreeClassifier()` in our model, and the hyperparamters will be `{'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 10}`. Later on, we contruct a Pipeline to train our data. 

The testing data overall performance of our Final Model is around 0.73. It is a huge inprovement comparing to our Baseline Model's score (0.58). Specifically, we include more features inside our Final Model, and the precision of our model is higher than the Baseline Model, since we contain only the datas that collecte from the first 15 minutes. This increase the complexity in a certain level for our model, which can better analyze the trend of the team during the game while not overfit to the training data. In addition, the training data's overall performance is around 0.76. This shows that our model does not occur overfitting condition since the prediction score on training and testing data only has a difference of 0.03

There is our confusion matrix for the testing data overall performance:

<iframe src="output2.png" width=800 height=600 frameBorder=0></iframe>


## Fairness Analysis
We split the data into groups - Group X, also called as short-games group, are games that last shorter than the 30 mins, which is around the 50% quartile. Group Y, also called a slong-games group, are games that last longer or equal to 30 mins. We weant to see if the precision score (the selected evaluation metric) of one group (shorter-games) is higher than the other (longer-games), in other words, if our final model perform worse in teams in long games than it does in teams in short games.

After computing the precision, recall and accuracy scores on these two groups using final model, we find out that they give almost the same scores on two groups. Since precision score has the largest gap, we choose it as our evaluation metric.

Our null hypothesis is that our model is fair. Its precision for long-game group and short-game group are roughly the same, and any differences are due to random chance. Our alternative hypothesis is that our model is unfair. Its precision for long-game group is lower than its precision for short-game group.

We choose 0.05 as our significance level. After predicting two groups' precision scores during permutation and comparing the experiment statistics with the observed statistic, the p-value is 0.0, which is below the significance level of 5%. Thus, we rejecet the null hypothesis. It has a great probability that the gap between two groups' precision scores are not due to random chance.
