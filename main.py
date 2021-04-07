import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sb
from termcolor import colored as cl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2

sb.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 10)

df = pd.read_csv('data/House_edited.csv')
# df.set_index("Id", inplace=True)

print(df.head(5))

df.dropna(inplace=True)

print(cl(df.isnull().sum(), attrs=['bold']))

print(df.describe())  # Get statistical view of the data such as mean, median, standard deviation...

print(cl(df.dtypes, attrs=['bold']))


def change_to_integers():
    items = ['mainroad', "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for item in items:
        df[item] = df[item].replace({"yes": 1, "no": 0})

    df['furnishingstatus'] = df['furnishingstatus'].replace({"furnished": 2, "unfurnished": 0, "semi-furnished": 1})

    df.to_csv("data/House_edited.csv")


sb.heatmap(df.corr(), annot=True, cmap='magma')
plt.savefig('data/heatmap.png')


def scatter_df(y_var):
    scatter_df = df.drop(y_var, axis=1)
    i = df.columns

    plot1 = sb.scatterplot(i[0], y_var, data=df, color='orange', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[0]), fontsize=16)
    plt.xlabel('{}'.format(i[0]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter1.png')

    plot2 = sb.scatterplot(i[1], y_var, data=df, color='yellow', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[1]), fontsize=16)
    plt.xlabel('{}'.format(i[1]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter2.png')

    plot3 = sb.scatterplot(i[2], y_var, data=df, color='aquamarine', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[2]), fontsize=16)
    plt.xlabel('{}'.format(i[2]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter3.png')

    plot4 = sb.scatterplot(i[3], y_var, data=df, color='deepskyblue', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[3]), fontsize=16)
    plt.xlabel('{}'.format(i[3]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter4.png')

    plot5 = sb.scatterplot(i[4], y_var, data=df, color='crimson', edgecolor='white', s=150)
    plt.title('{} / Sale Price'.format(i[4]), fontsize=16)
    plt.xlabel('{}'.format(i[4]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter5.png')

    plot6 = sb.scatterplot(i[5], y_var, data=df, color='darkviolet', edgecolor='white', s=150)
    plt.title('{} / Sale Price'.format(i[5]), fontsize=16)
    plt.xlabel('{}'.format(i[5]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter6.png')

    plot7 = sb.scatterplot(i[6], y_var, data=df, color='khaki', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[6]), fontsize=16)
    plt.xlabel('{}'.format(i[6]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter7.png')

    plot8 = sb.scatterplot(i[7], y_var, data=df, color='gold', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[7]), fontsize=16)
    plt.xlabel('{}'.format(i[7]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter8.png')

    plot9 = sb.scatterplot(i[8], y_var, data=df, color='r', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[8]), fontsize=16)
    plt.xlabel('{}'.format(i[8]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter9.png')

    plot10 = sb.scatterplot(i[9], y_var, data=df, color='deeppink', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[9]), fontsize=16)
    plt.xlabel('{}'.format(i[9]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter10.png')


# scatter_df('price')

# Distribution plot
def dist_plot():
    sb.distplot(df['price'], color='r')
    plt.title('Sale Price Distribution', fontsize=16)
    plt.xlabel('Sale Price', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('distplot.png')


def feature_sel_data_split():
    x_var = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "parking", "prefarea", "furnishingstatus"]]
    y_var = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.2, random_state=0)
    print(x_train.shape)
    return x_train, x_test, y_train, y_test


## Modeling
def model():
    x_train, x_test, y_train, y_test = feature_sel_data_split()
    model_1 = LinearRegression()
    model_1.fit(x_train, y_train)
    model_1_result = model_1.predict(x_test)
    model_1_file = open("data/linear_reg.model", 'wb')
    pickle.dump(model_1, model_1_file)
    model_1_file.close()

    ridge = Ridge(alpha=0.5)
    ridge.fit(x_train, y_train)
    ridge_result = ridge.predict(x_test)
    ridge_file = open("data/ridge.model", 'wb')
    pickle.dump(ridge, ridge_file)
    ridge_file.close()

    lasso = Lasso(alpha=0.01)
    lasso.fit(x_train, y_train)
    lasso_result = lasso.predict(x_test)
    lasso_file = open("data/lasso.model", 'wb')
    pickle.dump(lasso, lasso_file)
    lasso_file.close()

    bayesian = BayesianRidge()
    bayesian.fit(x_train, y_train)
    bayesian_result = bayesian.predict(x_test)
    bayesian_file = open("data/bayesian.model", 'wb')
    pickle.dump(bayesian, bayesian_file)
    bayesian_file.close()

    elastic = ElasticNet(alpha=0.01)
    elastic.fit(x_train, y_train)
    elastic_result = elastic.predict(x_test)
    elastic_file = open("data/elastic.model", 'wb')
    pickle.dump(elastic, elastic_file)
    elastic_file.close()

    return y_test, [model_1_result, ridge_result, elastic_result, lasso_result, bayesian_result]


y_test, results = model()
for result in results:
    score = evs(result, y_test)
    print(score)
