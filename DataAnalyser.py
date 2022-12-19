import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.svm import SVR


class DataAnalyser():
    TopChoice = 'Top choice'
    HighChoice = 'High choice'
    MidChoice = 'Middle choice'
    LowChoice = 'Low choice'
    ContestantWins = 'Contestant Wins'
    ChaserWins = 'Chaser Wins'
    difficulty = 0
    middle_offers = []
    differences_high = []
    differences_low = []
    info = []
    total = 288
    difficulty_any = 0
    difficulty_first = 0
    difficulty_second = 0
    difficulty_third = 0
    difficulty_fourth = 0

    def read_file(self, file):
        data = pd.read_csv(file)
        return data

    def data_plotter(self,data):
        data.loc[data['choice'] == self.TopChoice, 'picked'] = 1
        data.loc[data['choice'] == self.MidChoice, 'picked'] = 2
        data.loc[data['choice'] == self.LowChoice, 'picked'] = 3
        data.plot(kind='scatter', x='picked', y='wins_to_that_point', color='red')
        plt.show()
        plt.xlabel("wins_to_that_point")
        plt.ylabel("picked")

    def high_vs_mid(self, data):
        high_diff = data['high'] - data['mid']
        mid_offer = data['mid']
        df = pd.concat([mid_offer, high_diff], axis=1)
        # print(df.columns)
        # print(df)
        df.plot.scatter('mid', 0)
        plt.xlabel("Middle Offers")
        plt.ylabel("Difference between High Offers & Middle Offers")
        plt.title("Difference between Middle Offers and (High-Middle) Offers")
        plt.xticks(df['mid'].drop_duplicates())
        plt.show()

    def low_vs_mid(self, data):
        low_diff = data['mid'] - data['low']
        mid_offer = data['mid']
        df = pd.concat([mid_offer, low_diff], axis=1)
        # print(df.columns)
        # print(df)
        df.plot.scatter('mid', 0)
        plt.xlabel("Middle Offers")
        plt.ylabel("Difference between High Offers & Middle Offers")
        plt.title("Difference between Middle Offers and (Middle-Low) Offers")

        check = data.loc[(data['mid'] < data['low'])]
        print(check)

        plt.xticks(df['mid'].drop_duplicates())
        plt.show()

    def find_order_win_probability(self,data):
        info = (data.loc[(data['order'] == 1) & (data['win'] == self.ContestantWins)])
        size = info.shape[0]
        info1 = (data.loc[(data['order'] == 1)])
        size1 = info1.shape[0]
        percentage = size / size1 * 100
        print('order 1 win {:.2f} %'.format(percentage))
        info = (data.loc[(data['order'] == 2) & (data['win'] == self.ContestantWins)])
        size2 = info.shape[0]
        info1 = (data.loc[(data['order'] == 2)])
        size3 = info1.shape[0]
        percentage1 = size2 / size3 * 100
        print('order 2 win {:.2f} %'.format(percentage1))
        info = (data.loc[(data['order'] == 3) & (data['win'] == self.ContestantWins)])
        size4 = info.shape[0]
        info1 = (data.loc[(data['order'] == 3)])
        size5 = info1.shape[0]
        percentage2 = size4 / size5 * 100
        print('order 3 win {:.2f} %'.format(percentage2))
        info = (data.loc[(data['order'] == 4) & (data['win'] == self.ContestantWins)])
        size6 = info.shape[0]
        info1 = (data.loc[(data['order'] == 4)])
        size7 = info1.shape[0]
        percentage3 = size6 / size7 * 100
        print('order 4 win {:.2f} %'.format(percentage3))

    def find_mid_win_probability(self, data):
        #find probability of ANY contestant winning if someone takes the middle offer
        info = (data.loc[(data['choice'] == self.MidChoice) & (data['win'] == self.ContestantWins)])
        size = info.shape[0]
        info1 = (data.loc[(data['choice'] == self.MidChoice)])
        size1 = info1.shape[0]
        percentage = size / size1 * 100
        print('mid-choice win {:.2f} %'.format(percentage))

    def find_high_win_probability(self, data):
        #find probability of ANY contestant winning if someone takes the middle offer
        info = (data.loc[(data['choice'] == self.TopChoice) & (data['win'] == self.ContestantWins)])
        size = info.shape[0]
        info1 = (data.loc[(data['choice'] == self.TopChoice)])
        size1 = info1.shape[0]
        percentage = size / size1 * 100
        print('high-choice win {:.2f} %'.format(percentage))

    def find_low_win_probability(self, data):
        info1 = (data.loc[(data['choice'] == self.LowChoice)])
        size1 = info1.shape[0]
        #find probability of ANY contestant winning if someone takes the middle offer
        info = (data.loc[(data['choice'] == self.LowChoice) & (data['win'] == self.ContestantWins)])
        size = info.shape[0]
        percentage = size / size1 * 100
        print('low-choice win {:.2f} %'.format(percentage))




    def find_mid_win_probability_order(self, data, order):
        #find probability of a contestant winning, if someone takes the middle offer, depending on their order
        info = (data.loc[(data['choice'] == self.MidChoice) &
                         (data['win'] == self.ContestantWins) &
                         (data['order'] == order)])
        size = info.shape[0]
        info1 = (data.loc[(data['choice'] == self.MidChoice) & (data['order'] == order)])
        size1 = info1.shape[0]
        if size1 == 0:
            print('Chances of winning if order is {} and they pick the middle choice:{:.2f}% contestants'.format(order, 0))
            return 0
        else:
            percentage = size / size1 * 100
            print('Chances of winning if order is {} and they pick the middle choice:{:.2f}% out of {} contestants'.format(order,percentage, size1))

    def find_low_win_probability_order(self, data, order):
        #find probability of a contestant winning, if someone takes the low offer, depending on their order
        info = (data.loc[(data['choice'] == self.LowChoice) &
                         (data['win'] == self.ContestantWins) &
                         (data['order'] == order)])
        size = info.shape[0]
        info1 = (data.loc[(data['choice'] == self.LowChoice) & (data['order'] == order)])
        size1 = info1.shape[0]
        if size1 == 0:
            print('Chances of winning if order is {} and they pick the low choice:{:.2f}% out of {} contestants'.format(order, 0, size1))
        else:
            percentage = size / size1 * 100
            print('Chances of winning if order is {} and they pick the low choice:{:.2f}% out of {} contestants'.format(order,percentage, size1))

    def find_high_win_probability_order(self, data, order):
        #find probability of a contestant winning, if someone takes the middle offer, depending on their order
        info = (data.loc[(data['choice'] == self.TopChoice) &
                         (data['win'] == self.ContestantWins) &
                         (data['order'] == order)])
        size = info.shape[0]
        info1 = (data.loc[(data['choice'] == self.TopChoice) & (data['order'] == order)])
        size1 = info1.shape[0]
        if size1 == 0:
            print('Chances of winning if order is {} and they pick the high choice:{:.2f}% out of {} contestants'.format(order, 0, size1))
        else:
            percentage = size / size1 * 100
            print('Chances of winning if order is {} and they pick the high choice:{:.2f}% out of {} contestants'.format(order,percentage, size1))

    def regression(self, data):
        data.loc[data['choice'] == self.TopChoice, 'picked'] = data['high']
        data.loc[data['choice'] == self.MidChoice, 'picked'] = data['mid']
        data.loc[data['choice'] == self.LowChoice, 'picked'] = data['low']
        data = data.assign(questions_correct=lambda x: (x['mid'] / 2000))
        # print(data)
        print(data['picked'])
        offers = data[["picked","questions_correct","high","mid","low","order","wins_to_that_point", "fund"]]
        # regression to check whether the number of questions gotten correct in cash-raise round + order of contestant
        # + number of contestants left, has an effect on the offer picked
        result = sm.ols(formula="picked ~  high + mid + low + order + wins_to_that_point + fund", data=offers).fit()
        print(result.summary())

    def logistic_regression(self,data):
         data.loc[data['choice'] == self.TopChoice, 'picked'] = data['high']
         data.loc[data['choice'] == self.MidChoice, 'picked'] = data['mid']
         data.loc[data['choice'] == self.LowChoice, 'picked'] = data['low']
         data = data.assign(questions_correct=lambda x: (x['mid'] / 2000))
         offers = data[["picked", "high", "mid","low", "order", "wins_to_that_point", "fund"]]
         offers = offers.dropna()
         offers.reset_index()
         data_y = offers['picked']
         data_X = offers.drop(['picked'], axis=1)
         X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                             random_state=123, test_size=0.2)
         scaler = StandardScaler().fit(X_train)
         train_scaled = pd.DataFrame(scaler.transform(X_train))
         test_scaled = pd.DataFrame(scaler.transform(X_test))

         #unbalanced logistic regression
         rg = LogisticRegression(penalty='l2', class_weight=None, max_iter=1000).fit(train_scaled, y_train)
         y_pred0 = rg.predict(test_scaled)
         coef = rg.coef_[0]
         print('coef', coef)

         # print('unbalanced ', accuracy_score(y_test, y_pred0))
         # print(clf.predict_proba(train_scaled)[0])
         # print(clf.predict(train_scaled)[0])
         # print('regression')
         self.find_errors(y_test, y_pred0)

    def random_forest_regression(self, data):
        data.loc[data['choice'] == self.TopChoice, 'picked'] = data['high']
        data.loc[data['choice'] == self.MidChoice, 'picked'] = data['mid']
        data.loc[data['choice'] == self.LowChoice, 'picked'] = data['low']
        data = data.assign(questions_correct=lambda x: (x['mid'] / 2000))
        offers = data[["picked", "high", "mid", "low", "order", "wins_to_that_point", "fund"]]
        offers = offers.dropna()
        offers.reset_index()
        data_y = offers['picked']
        data_X = offers.drop(['picked'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                            random_state=123, test_size=0.2)
        scaler = StandardScaler().fit(X_train)
        train_scaled = pd.DataFrame(scaler.transform(X_train))
        test_scaled = pd.DataFrame(scaler.transform(X_test))

        #predicts real valued outputs which vary and don't require outputs
        #predicted to be in a fixed set
        rfr = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True, max_depth=25)
        rfr.fit(train_scaled, y_train)
        y_pred0 = rfr.predict(test_scaled)
        print(y_pred0)
        print(pd.DataFrame({'Variable': train_scaled.columns,
                      'Importance': rfr.feature_importances_}).sort_values('Importance', ascending=False))
        print('rfr')
        self.find_errors(y_test, y_pred0)

        #predicts a set of specified labels (1,2,3)
        # forest = RandomForestClassifier()
        # forest.fit(train_scaled, y_train)
        # y_pred_test = forest.predict(test_scaled)
        # print(y_pred_test)
        # print(accuracy_score(y_test, y_pred_test))
        # print(classification_report(y_test, y_pred_test))

    def svm_regression(self, data):
        N = 9
        # Select first N columns
        first_n_column = data.iloc[:, :N]
        print(first_n_column)
        data.loc[data['choice'] == self.TopChoice, 'picked'] = data['high']
        data.loc[data['choice'] == self.MidChoice, 'picked'] = data['mid']
        data.loc[data['choice'] == self.LowChoice, 'picked'] = data['low']
        data = data.assign(questions_correct=lambda x: (x['mid'] / 2000))
        offers = data[["picked", "high", "mid", "low", "order", "wins_to_that_point","fund"]]
        offers = offers.dropna()
        offers.reset_index()
        data_y = offers['picked']
        data_X = offers.drop(['picked'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                            random_state=123, test_size=0.2)
        scaler = StandardScaler().fit(X_train)
        train_scaled = pd.DataFrame(scaler.transform(X_train))
        test_scaled = pd.DataFrame(scaler.transform(X_test))
        svm = SVR(kernel = 'linear')
        svm.fit(train_scaled, y_train)
        y_pred0 = svm.predict(test_scaled)
        print(test_scaled)
        print(y_pred0)
        print(svm.coef_)
        # print('svm')
        self.find_errors(y_test, y_pred0)
        print(pd.DataFrame({'Variable': train_scaled.columns,
                      'Importance': svm.feature_importances_}).sort_values('Importance', ascending=False))

    def svm_regression_poly(self, data):
        data.loc[data['choice'] == self.TopChoice, 'picked'] = data['high']
        data.loc[data['choice'] == self.MidChoice, 'picked'] = data['mid']
        data.loc[data['choice'] == self.LowChoice, 'picked'] = data['low']
        data = data.assign(questions_correct=lambda x: (x['mid'] / 2000))
        offers = data[["picked", "high", "mid", "low", "order", "wins_to_that_point","fund"]]
        offers = offers.dropna()
        offers.reset_index()
        data_y = offers['picked']
        data_X = offers.drop(['picked'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,random_state=123, test_size=0.2)
        degree = 2
        poly = PolynomialFeatures(degree)
        expanded_train = poly.fit_transform(X_train)
        expanded_test = poly.fit_transform(X_test)
        # Verify the number of dimensions.
        print(poly.n_output_features_)
        print(poly.get_feature_names(data_X.columns))
        scaler = StandardScaler().fit(expanded_train)
        train_scaled = pd.DataFrame(scaler.transform(expanded_train))
        test_scaled = pd.DataFrame(scaler.transform(expanded_test))
        svm = SVR(kernel = 'linear')
        svm.fit(train_scaled, y_train)
        y_pred0 = svm.predict(test_scaled)
        print(svm.coef_)
        self.find_errors(y_test, y_pred0)

    def find_errors(self, y_test, y_pred0):
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred0))
        print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred0))
        print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_test, y_pred0, squared=False))
        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_test, y_pred0))
        print('Explained Variance Score:', metrics.explained_variance_score(y_test, y_pred0))
        print('Max Error:', metrics.max_error(y_test, y_pred0))
        print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred0))
        print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred0))
        print('R^2:', metrics.r2_score(y_test, y_pred0))
        print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_test, y_pred0))
        print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_test, y_pred0))

    def update_pot(self, data):
        # we want to find the total money in the prize fund when a contestant goes up
        # we have contestant order
        # cont 1 = 0
        # cont 2 = add money if cont 1 wins
        # cont 3 = add money if cont 2 wins
        # cont 4 = add money if cont 3 wins
        data.insert(9,'fund',0)
        for index, row in data.iterrows():
            if row['order'] == 1:
                data.at[index, 'fund'] = 0
            else:
                if data.at[index-1,'win'] == self.ContestantWins:
                    if data.at[index-1,'choice'] == self.MidChoice:
                        data.at[index, 'fund'] = data.at[index-1, 'fund'] + data.at[index-1, 'mid']
                    elif data.at[index-1,'choice'] == self.LowChoice:
                        data.at[index, 'fund'] = data.at[index-1, 'fund'] + data.at[index-1, 'low']
                    else:
                        data.at[index, 'fund'] = data.at[index - 1, 'fund'] + data.at[index - 1, 'high']
        data.to_csv('information.txt')

if __name__ == "__main__":
    x = DataAnalyser()
    data = x.read_file('information.csv')
    #plotter vs visualisations
    # x.high_vs_mid(data)
    # x.low_vs_mid(data)
    # x.data_plotter(data)

    #general info about wins according choices
    # x.find_mid_win_probability(data)
    # x.find_low_win_probability(data)
    # x.find_high_win_probability(data)
    # x.find_order_win_probability(data)

    #general info about wins according to order & choices
    # for i in range(1,5):
    #     x.find_mid_win_probability_order(data,i)
    #     x.find_low_win_probability_order(data,i)
    #     x.find_high_win_probability_order(data,i)

    #data analysis
    x.regression(data)
    x.logistic_regression(data)
    x.random_forest_regression(data)
    x.svm_regression(data)
    x.svm_regression_poly(data)