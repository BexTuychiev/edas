import streamlit as st
from PIL import Image

st.markdown("""# My 5-part Powerful EDA Template That Speaks of Ultimate Skill
## It is hard to stand out guys...""")

st.markdown("""
<img src='https://cdn-images-1.medium.com/max/1200/1*JwZ7l0LNnIVP9bPNjI8KZQ.jpeg' width=800></img>
<figcaption style="text-align: center;">
    <strong>
        Photo by 
        <a href='https://pixabay.com/users/gam-ol-2829280/?utm_source=link-attribution&utm_medium=referral&utm_campaign=
        image&utm_content=3319946'>Oleg Gamulinskiy</a>
        on 
        <a href='https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=
        3319946'>Pixabay</a>
    </strong>
</figcaption>""", unsafe_allow_html=True)

st.markdown("""
# 1. Introduction to the Dataset And The Aim of the EDA
Today, more and more businesses are offering their services through mobile applications. Because of this massive number of apps it is hard to compete and get your product in front of potential customers. So, it is important to make thorough research into your niche category and find out how you would stack up against your competitors. 
This [dataset](https://www.kaggle.com/gauthamp10/google-playstore-apps) of more than 1 million apps on the Google Play Store can be used for exactly that purpose. This notebook tries to compare the performance of free and paid apps in the top 8 most popular app categories on Google Play Store (excluding games):
- Education
- Business
- Music & Audio
- Tools
- Entertainment
- Lifestyle
- Books & Reference
- Food & Drink

## 1.1 Library Setup and Read in the Data
""")
with st.echo():
    # Base libraries
    import time
    import datetime
    import os

    # Scientific libraries
    import numpy as np
    import pandas as pd
    from empiricaldist import Cdf, Pmf

    # Visual libraries
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import seaborn as sns
    import missingno as msno  # Visualize missing values

    # Helper libraries
    from tqdm.notebook import tqdm, trange
    from colorama import Fore, Back, Style
    import warnings

    warnings.filterwarnings('ignore')

    # Visual setup
    import matplotlib.ticker as ticker

    plt.style.use('ggplot')
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['figure.figsize'] = [12, 9]
    rcParams['font.size'] = 16
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    custom_colors = ['#74a09e', '#86c1b2', '#98e2c6', '#f3c969',
                     '#f2a553', '#d96548', '#c14953']
    sns.set_palette(custom_colors)

    # Pandas options
    pd.set_option('max_colwidth', 40)
    pd.options.display.max_columns = None  # Possible to limit
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.ast_node_interactivity = 'all'

    # Seed value for numpy.random
    np.random.seed(42)
apps = pd.read_csv('google_play_store/data/Google-Playstore.csv')
with st.echo():
    apps.info()
with st.echo():
    print(apps.info())
apps.rename(lambda x: x.lower().strip().replace(' ', '_'),
            axis='columns', inplace=True)

# Specify the cols to drop
to_drop = [
    'app_id', 'minimum_android',
    'developer_id', 'developer_website', 'developer_email', 'privacy_policy',
    'ad_supported', 'in_app_purchases', 'editors_choice'
]

# Drop them
apps.drop(to_drop, axis='columns', inplace=True)

# Collapse 'Music' and 'Music & Audio' into 'Music'
apps['category'] = apps['category'].str.replace('Music & Audio', 'Music')

# Collapse 'Educational' and 'Education' into 'Education'
apps['category'] = apps['category'].str.replace('Educational', 'Education')

top_8_list = [
    'Education', 'Music', 'Business', 'Tools',
    'Entertainment', 'Lifestyle', 'Food & Drink',
    'Books & Reference'
]

top = apps[apps['category'].isin(top_8_list)].reset_index(drop=True)

# Specifying the datetime format significantly reduces conversion time
top['released'] = pd.to_datetime(top['released'], format='%b %d, %Y',
                                 infer_datetime_format=True, errors='coerce')

st.markdown('Load in the data:')

st.code("""apps = pd.read_csv('../google_play_store/input/google-playstore-apps/Google-Playstore.csv')
apps.head()""")

st.dataframe(apps.head())

st.markdown("""
# 2. Basic Exploration and Data Cleaning
## 2.1 Basic Exploration
We will start with basic exploration of the dataset and get a feel for how it looks.
""")

st.code("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1118136 entries, 0 to 1118135
Data columns (total 23 columns):
 #   Column             Non-Null Count    Dtype
---  ------             --------------    -----
 0   App Name           1118135 non-null  object
 1   App Id             1118136 non-null  object
 2   Category           1118133 non-null  object
 3   Rating             1111286 non-null  float64
 4   Rating Count       1111286 non-null  float64
 5   Installs           1117975 non-null  object
 6   Minimum Installs   1117975 non-null  float64
 7   Maximum Installs   1118136 non-null  int64
 8   Free               1118136 non-null  bool
 9   Price              1118136 non-null  float64
 10  Currency           1117975 non-null  object
 11  Size               1118136 non-null  object
 12  Minimum Android    1116123 non-null  object
 13  Developer Id       1118134 non-null  object
 14  Developer Website  703770 non-null   object
 15  Developer Email    1118114 non-null  object
 16  Released           1110406 non-null  object
 17  Last Updated       1118136 non-null  object
 18  Content Rating     1118136 non-null  object
 19  Privacy Policy     964612 non-null   object
 20  Ad Supported       1118136 non-null  bool
 21  In App Purchases   1118136 non-null  bool
 22  Editors Choice     1118136 non-null  bool
dtypes: bool(4), float64(4), int64(1), object(14)
memory usage: 166.3+ MB """)

st.markdown("""
There are 23 columns and several of them have missing values. I like creating a single cell to list all the issues that need addressing and deal with them separately after I am done exploring. Then, I can cross each issue one by one as I fix them.
""")

st.markdown("""
I like the columns of my dataset to have *snake_case* because it will be easier to choose them later on (added to the list).

Also, I think these columns will not be useful for us: App ID, minimum and maximum installs, minimum android version, developer ID, website and email, privacy policy link.

Next, looking at the dataset info again:
""")

st.code("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1118136 entries, 0 to 1118135
Data columns (total 23 columns):
 #   Column             Non-Null Count    Dtype
---  ------             --------------    -----
 0   App Name           1118135 non-null  object
 1   App Id             1118136 non-null  object
 2   Category           1118133 non-null  object
 3   Rating             1111286 non-null  float64
 4   Rating Count       1111286 non-null  float64
 5   Installs           1117975 non-null  object
 6   Minimum Installs   1117975 non-null  float64
 7   Maximum Installs   1118136 non-null  int64
 8   Free               1118136 non-null  bool
 9   Price              1118136 non-null  float64
 10  Currency           1117975 non-null  object
 11  Size               1118136 non-null  object
 12  Minimum Android    1116123 non-null  object
 13  Developer Id       1118134 non-null  object
 14  Developer Website  703770 non-null   object
 15  Developer Email    1118114 non-null  object
 16  Released           1110406 non-null  object
 17  Last Updated       1118136 non-null  object
 18  Content Rating     1118136 non-null  object
 19  Privacy Policy     964612 non-null   object
 20  Ad Supported       1118136 non-null  bool
 21  In App Purchases   1118136 non-null  bool
 22  Editors Choice     1118136 non-null  bool
dtypes: bool(4), float64(4), int64(1), object(14)
memory usage: 166.3+ MB
""")

st.markdown("""
Some columns have incorrect data types: Released, Size. Released should be a `datetime`. Size is probably rendered as string because each size contains the letter 'M' to indicate megabytes. These issues will be added to the list too.

Now, let's look at the app categories:
""")

st.code("""print(apps['Category'].value_counts())""")

st.code("""Education                  115242
Music & Audio              104541
Entertainment               82079
Books & Reference           78969
Personalization             73538
Tools                       68953
Lifestyle                   54586
Business                    42210
Health & Fitness            31532
Productivity                30450
Photography                 28942
Travel & Local              25971
Puzzle                      24954
Finance                     24846
Food & Drink                24250
Sports                      22094
News & Magazines            21553
Casual                      20509
Shopping                    20440
Communication               18391
Arcade                      17715
Social                      16987
Simulation                  15372
Action                      12628
Medical                     12554
Art & Design                12322
Educational                 11351
Maps & Navigation           10468
Adventure                   10124
Video Players & Editors      9095
Auto & Vehicles              6872
Beauty                       6236
Racing                       6004
Role Playing                 5632
House & Home                 5475
Trivia                       5470
Board                        5261
Word                         4677
Card                         4674
Strategy                     4071
Events                       3788
Weather                      2958
Dating                       2883
Casino                       2648
Music                        2515
Libraries & Demo             2382
Comics                       2137
Parenting                    1784
Name: Category, dtype: int64""")

st.markdown("""
If we look carefully, some categories of interest like Music and Eduction are given with different labels: there are both 'Music & Audio' and 'Music' labels as well as 'Education' and 'Educational' for education. They should be merged together to represent a single category. 

Later, we will subset for the top 8 columns after finishing cleaning.

Now, let's explore the numerical features of the dataset and see if they contain any issues:""")

st.code("""
    # Display in normal notation instead of scientific
    with pd.option_context('float_format', '{:f}'.format):
        print(apps.describe())
""")

st.code("""
              Rating     Rating Count   Minimum Installs   Maximum Installs  
count 1111286.000000   1111286.000000     1117975.000000     1118136.000000
mean        2.490334      5159.633249      313643.231397      544453.372541
std         2.053973    272409.445115    20439406.354105    30310580.652237
min         0.000000         0.000000           0.000000           0.000000
25%         0.000000         0.000000         100.000000         160.000000
50%         3.600000        11.000000        1000.000000        1719.000000
75%         4.300000       100.000000       10000.000000       19116.000000
max         5.000000 125380770.000000 10000000000.000000 10772700105.000000

               Price
count 1118136.000000
mean        0.205073
std         3.541011
min         0.000000
25%         0.000000
50%         0.000000
75%         0.000000
max       400.000000
""")

st.markdown("""
Looks like all numerical columns are within the sensible range, like rating should be between 0 and 5. But the maximum value for price is 400$ which is a bit suspicious. We will dig into that later.

Before we further explore, let's deal with the issues we highlighted. Here is the final list:
""")

st.markdown("""
### Issues List For the Dataset
- Missing values in several cols: Rating, rating count, Installs, minimum and maximum installs, currency and more
- Convert all columns to snake_case
- Drop these columns: App ID, minimum and maximum installs, minimum android version, developer ID, website and email, privacy policy link.
- Incorrect data types for release data and size
- Music and education is represented by different labels
- Drop unnecessary categories
""")

st.markdown("""
## 2.2 Data Cleaning
It is a good practice to start cleaning from the easiest issues.
""")

st.markdown("""
### Convert all columns to snake case
""")

st.code("""
apps.rename(lambda x: x.lower().strip().replace(' ', '_'),
            axis='columns', inplace=True)
""")

st.markdown("""
Check the results:
""")

st.code("""print(apps.columns)""")

st.code("""
Index(['app_name', 'app_id', 'category', 'rating', 'rating_count', 'installs',
       'minimum_installs', 'maximum_installs', 'free', 'price', 'currency',
       'size', 'minimum_android', 'developer_id', 'developer_website',
       'developer_email', 'released', 'last_updated', 'content_rating',
       'privacy_policy', 'ad_supported', 'in_app_purchases', 'editors_choice'],
      dtype='object')
""")

st.markdown("""
### Drop unnecessary columns
""")

st.code("""
    # Specify the cols to drop
    to_drop = [
        'app_id', 'minimum_android',
        'developer_id', 'developer_website', 'developer_email', 'privacy_policy',
        'ad_supported', 'in_app_purchases', 'editors_choice'
    ]

    # Drop them
    apps.drop(to_drop, axis='columns', inplace=True)
""")

st.markdown("""Check:""")

st.code("""assert apps.columns.all() not in to_drop""")

st.markdown("""### Collapse multiple categories into one""")

st.code("""    # Collapse 'Music' and 'Music & Audio' into 'Music'
    apps['category'] = apps['category'].str.replace('Music & Audio', 'Music')

    # Collapse 'Educational' and 'Education' into 'Education'
    apps['category'] = apps['category'].str.replace('Educational', 'Education')
""")

st.markdown("""Check:""")

st.code("""assert 'Educational' not in apps['category'] and \
           'Music & Audio' not in apps['category']""")

st.markdown("""### Subset only for top 8 categories""")

st.code("""    top_8_list = [
        'Education', 'Music', 'Business', 'Tools',
        'Entertainment', 'Lifestyle', 'Food & Drink',
        'Books & Reference'
    ]

    top = apps[apps['category'].isin(top_8_list)].reset_index(drop=True)
""")


st.markdown("""Check:""")

st.code("""assert top['category'].all() in top_8_list""")

st.markdown("""### Convert `released` to `datetime`""")

st.code("""    # Specifying the datetime format significantly reduces conversion time
    top['released'] = pd.to_datetime(top['released'], format='%b %d, %Y',
                                     infer_datetime_format=True, errors='coerce')
""")

st.markdown("""Check:""")

st.code("""print(top.released.dtype)""")

st.code("""dtype('<M8[ns]')""")

st.markdown("""### Convert `size` to float""")

st.code("""    # Strip of all text and convert to numeric
    top['size'] = pd.to_numeric(top['size'].str.replace(r'[a-zA-Z]+', ''),
                                errors='coerce')
""")

st.markdown("""Check:""")

st.code("""assert top['size'].dtype == 'float64'""")

st.markdown("""No output means passed!""")

st.markdown("""
### Deal With Missing Values

There seems to be much more missing values in `size`, well over the threshold where we can safely drop them. Let's dive deeper using the `missingno` package:
""")

with st.echo():
    fig, ax = plt.subplots()
    msno.matrix(top.sort_values('category'), ax=ax)

st.pyplot(fig)

st.markdown("""> If you don't understand this plot or what I just did here, read my [article](
https://towardsdatascience.com/how-to-identify-missingness-types-with-missingno-61cfe0449ad9) on the topic. 

Plotting the data sorted by category tells that nulls in `size` are randomly scattered. Let's plot the missingness correlation:
""")

with st.echo():
    fig, ax = plt.subplots()
    msno.heatmap(top, cmap='rainbow', ax=ax)

st.pyplot(fig)

st.markdown("""The correlation matrix of missingness shows that most nulls in `size` are associated with nulls in 
`rating` and `rating_count`. This scenario falls into the Missing At Random class of missingness. This means that 
even though null values are random their misssingness is related to other observed values. 

In most cases, you can impute them with ML techniques. But our case is specific because we don't have enough features that help predict the size or rating of an app. We have no choice but to drop them. (I am not even mentioning nulls in other columns because their proportion is marginal)
""")

with st.echo():
    top.dropna(inplace=True)

st.markdown("""# 3. Univariate Exploration""")

st.markdown("""Finally, time to create some plots. Specifically, we will look at the distribution of the numerical 
features of the dataset. 

Let's start with rating:""")

with st.echo():
    fig, ax = plt.subplots()

    # Plot a histogram
    ax.hist(top['rating'], bins=20)
    # Label
    ax.set(title='A Histogram of App ratings',
           xlabel='Rating out of 5.0',
           ylabel='Count')
    plt.show()

st.pyplot(fig)

st.markdown("""
It looks like, there are much more apps with no rating. We might get a better visual if omit them:
""")

with st.echo():
    fig, ax = plt.subplots()

    # Subset for ratings over 0
    over_0 = top[top['rating'] > 0]['rating']

    # Plot a histogram
    ax.hist(over_0, bins=15)

    # Label
    ax.set(title='A Histogram of App ratings',
           xlabel='Rating out of 5.0',
           ylabel='Count')

    plt.show()

st.pyplot(fig)

st.markdown("""
Histogram shows that majority of the apps are rated between ~3.8 and 4.8. It is also surprising to see so many 5-star ratings.
""")

st.markdown("""
Now, let's look at how categories are distributed:
""")

with st.echo():
    fig, ax = plt.subplots()

    # Plot a normalized couplet
    top['category'].value_counts(normalize=True).plot.barh()

    # Label
    ax.set(title='Proportion of 8 Categories',
           xlabel='Proportion', ylabel='')

    plt.show()

st.pyplot(fig)

st.markdown("""
Looks like educational apps make up more than one fifth of the data. 

It would be ideal if we had the `install_count` given as integers. But they are collapsed into categories which makes it impossible to see their distribution as a numeric feature. We got no other choice but to plot them in the same way as above:
""")

with st.echo():
    fig, ax = plt.subplots()

    # Create a normalized countplot
    top.installs.value_counts(normalize=True).plot.barh()

    ax.set(title='Proportion of install categories',
           xlabel='Proportion', ylabel='')

    plt.show()

st.pyplot(fig)

st.markdown("""
The plot shows that the vast majority of installs are between 10 and 10k installs. Maybe, we could get a better insight if we plotted `rating_count`. The number of ratings is given as exact figures and logically, they are positively related to install count. Before plotting, let's get the 5-number-summary of `rating_count`:
""")

with st.echo():
    with pd.option_context('float_format', '{:f}'.format):
        top.rating_count.describe()

st.code("""
count     570129.000000
mean        1304.621249
std        42618.451999
min            0.000000
25%            0.000000
50%            8.000000
75%           59.000000
max     16802391.000000
Name: rating_count, dtype: float64
""")

st.markdown("""
WOW! From the summary we can see that 75% of the distribution is less than 59 while the max is over 1.6 million. So, we will only plot the apps with ratings <60:""")

with st.echo():
    fig, ax = plt.subplots()

    # Choose apps with ratings <60
    percentile_75 = top[top['rating_count'] < 60]

    ax.hist(percentile_75['rating_count'], bins=15)

    ax.set(title='Histogram of Rating Count',
           xlabel='Ratings', ylabel='Count')

    plt.show()

st.pyplot(fig)

st.markdown("""
This histogram tells us that about a quarter of the apps have no more than 5 ratings. This shows how competetive the mobile market is. Only a small proportion of apps can go as popular as the ones with thousands of ratings. Just for curiosity, let's explore the apps that have more than 1 million ratings:
""")

with st.echo():
    over_mln = top[top['rating_count'] > 1e6]
    print(over_mln.shape)

st.code("(76, 14)")

st.markdown("""
Out of the initial 1 million apps, only 76 have over 1 million ratings. Let's see which categories have the most number of apps:
""")

with st.echo():
    fig, ax = plt.subplots()

    over_mln.category.value_counts().plot.barh()

    ax.set(title='Comparison of Rating Count of Most Popular Apps by Category',
           ylabel='Rating Count')

    plt.show()

st.pyplot(fig)

st.markdown("""
Not surprisingly, 30 of the apps belong to `Tools` which probably include most popular everyday apps.

Finally, let's look at the 5-number summary of the price of paid apps:
""")

with st.echo():
    # Create a mask for paid apps
    is_paid = top['price'] != 0

    with pd.option_context('float_format', '{:f}'.format):
        top[is_paid]['price'].describe()

st.code("""
count   20072.000000
mean        5.421512
std        16.976231
min         0.990000
25%         0.990000
50%         2.490000
75%         4.990000
max       399.990000
Name: price, dtype: float64
""")

st.markdown("""
Again, `price` also contains some serious outliers. For best insights, we subset for apps that cost less than 10$:
""")

with st.echo():
    fig, ax = plt.subplots()

    # Subset for apps that cost less than 10$
    less_10 = top[(top['price'] > 0) & (top['price'] < 10)]

    # Create a PMF of price for apps
    pmf_price = Pmf.from_seq(less_10['price'])

    pmf_price.bar()

    ax.set(title='PMF of Price of Paid Apps',
           xlabel='Price ($)', ylabel='$P(X = x)$')

    plt.show()

st.pyplot(fig)

st.markdown("""
> I used a Probability Mass Function to compare the distributions. I find them much better than historgrams. Because, they avoid binning bias of histograms and allows for improved visual comparison of distributions. Learn more about them [here](https://towardsdatascience.com/3-best-often-better-histogram-alternatives-avoid-binning-bias-1ffd3c811a31?source=your_stories_page-------------------------------------).
""")

st.markdown("""
It is clear that most apps cost about a dollar.""")

st.markdown("""
# 4. Bivariate Exploration
Coming to bivariate exploration, we will start by comparing the distributions among categories:
""")

with st.echo():
    fig, ax = plt.subplots()
    # Extract the unique categories
    categories = top['category'].unique()
    # Filter out 0-star ratings
    over_0 = top[top['rating'] > 0]

    for cat in categories:
        pmf_cat = Pmf.from_seq(over_0[over_0['category'] == cat]['rating'])
        ax.plot(pmf_cat, label=cat)

    ax.set(title='PMF of Rating For Top 8 App Categories',
           xlabel='Rating',
           ylabel='P(X = x)')

    ax.legend()

    plt.show()

st.pyplot(fig)

st.markdown("""
Probability Mass Function of app ratings show that the distributions of rating for top 8 are pretty similar. However, more apps in Music and Tools seem to be rated ~4.6 compared to other groups.

From this, we can also conclude that rating counts and number of installs probably have similar distributions across categories.

What would be interesting is to compare prices of the apps across categories. I am guessing that apps for business will be the most expensive but let's make sure with a proper visual:
""")

with st.echo():
    fig, ax = plt.subplots()

    sns.boxplot(x='category', y='price', data=top[is_paid])

    ax.set_yscale('log')

    ax.set(title='Comparison of Price Between Categories',
           xlabel='', ylabel='Price ($)')

    # Rotate xtick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    # Set custom yticklabels
    y_ticks = [0.3, 0.5, 1, 3, 5, 10, 30, 100, 300]
    ax.set_yticklabels(y_ticks)

    plt.show()

st.pyplot(fig)

st.markdown("""
Well, I was off by one category. Looking at medium price for each, `Food & Drink` seems to be the winner closely followed by `Business`.

Now, let's see if more ratings mean higher ratings. Again, we will only look at apps with ratings fewer than 100k and exclude the ones with no ratings:
""")

with st.echo():
    fig, ax = plt.subplots()

    # Filter out undesired apps
    majority_rated = top[(top['rating_count'] > 0) & (top['rating_count'] < 1e5)]

    # Jitter the ratings
    rating_jittered = majority_rated['rating'] + np.random.normal(0, 0.12, len(majority_rated))
    # Jitter the number of ratings
    count_jittered = majority_rated['rating_count'] + np.random.normal(0, 1, len(majority_rated))

    ax.plot(count_jittered, rating_jittered,
            marker='o', linestyle='none', markersize=1, alpha=0.05)

    # Use log scale
    ax.set_xscale('log')
    # Set custom tick labels
    x_tick_labels = [1, 10, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]
    ax.set_xticklabels(x_tick_labels)

    # Label
    ax.set(title='Rating Count vs. Rating',
           xlabel='Rating Count',
           ylabel='Rating')

    plt.show()

st.pyplot(fig)

st.markdown("""
Even though there are much more apps with few but high ratings, there seems to be a weak positive non-linear relationship between rating count and rating.

We can check this by plotting a correlation matrix:
""")

with st.echo():
    fig, ax = plt.subplots()

    sns.heatmap(top.corr(), annot=True,
                linewidths=3, center=0, cmap='rainbow', ax=ax)

st.pyplot(fig)

st.markdown("""
Correlation matrix does not tell us much. However, we could confirm our earlier notion that there is a non-linear positive relationship between rating and rating count with a coefficient of $r=0.028$. 

Now, let's explore other angles. We will look at content rating of the apps and grouping them into paid and free categories:
""")

with st.echo():
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.countplot(x='content_rating', hue='free', data=top,
                  palette=['#74a09e', '#f2a553'])

    ax.legend(['Paid', 'Free'], loc='upper right')

    ax.set(title='Comparison of Content Rating For Paid And Free Apps',
           xlabel='Content Rating', ylabel='Count')

    plt.show()

st.pyplot(fig)

st.markdown("""The countplot shows that the vast majority is rated for `Everyone`. """)

st.markdown("""
# 5. Multivariate Exploration

Honestly, at this point, I am little disappointed because the dataset did not turn out to be that interesting. To create more angles to explore, we will use the minimum and maximum installs columns. Looking at the dataset documentation, they show estimated minimum and maximum installs.

We will take the mean of the two columns which would be a better estimate for the true install count:
""")

with st.echo():
    top['true_installs'] = (top['minimum_installs'] + top['maximum_installs']) // 2

st.markdown("""
We will see if there is a relationship between rating and install count. We will also group the plots into paid and free apps.:
""")

with st.echo():
    top['free'] = top['free'].replace({True: 'Free', False: 'Paid'})

    fig, ax = plt.subplots()

    # Create a boxplot
    sns.boxplot(x='category', y='rating', hue='free', data=top[top['rating'] > 0],
                palette=['#74a09e', '#f2a553'])

    # Labelling
    ax.set(title='Comparison of Rating For Both Free And Paid Apps',
           xlabel='', ylabel='Rating')

    # Rotate xticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    ax.legend(title='')

    plt.show()

st.pyplot(fig)

st.markdown("""
A couple interesting things from the above plot: Education and Books categories have the same median rating. Also, Tools and Business apps are generally lower rated than other categories. But it seems that the fact that app is free or paid does not have any effect ont its rating. 

Lastly, we will look at if well-maintainted apps have higher ratings. To achieve this, we will look at the `last_updated` column and compare it to the date of the latest version of this dataset, which was December, 2020.

First, we will find out how many days have passed since an app was last updated:
""")

with st.echo():
    # Covert last_updated to datetime
    top['last_updated'] = pd.to_datetime(top['last_updated'], format='%b %d, %Y',
                                         infer_datetime_format=True)

    top['days_elapsed'] = (datetime.datetime(2020, 12, 31) - top['last_updated']).dt.days

st.markdown("""
Let's see 5-number summary of `days_elapsed`:
""")

with st.echo():
    top.days_elapsed.describe()

st.code("""
count    570129.000000
mean        499.277641
std         532.959011
min          28.000000
25%         118.000000
50%         309.000000
75%         689.000000
max        4004.000000
Name: days_elapsed, dtype: float64
""")

st.markdown("""
It seems most apps were updated within a year at the time of collecting this dataset. Let's plot it against `rating` grouping by `Free` or `Paid` (Again, exclude apps with no rating):
""")

with st.echo():
    fig, ax = plt.subplots()

    with_rating = top[top['rating'] > 0]

    # Jitter the ratings for a nicer visual
    with_rating['rating_jittered'] = with_rating['rating'] + np.random.normal(0, 0.12, len(with_rating))
    sns.scatterplot(x='days_elapsed', y='rating_jittered', hue='free', data=with_rating,
                    alpha=0.5, s=1, palette=['#74a09e', '#f2a553'])

    ax.set(title='Maintenance of App vs. Rating', xlabel='Days Elapsed Since Last Updated',
           ylabel='Rating')

    ax.set_xscale('log')
    ax.set_xticklabels([10, 30, 70, 100, 300, 700, 1000])
    ax.legend(title='')
    plt.show()

st.pyplot(fig)

st.markdown("""
It seems maintenance didn't have any effect on rating. But we should also conside the possibility that not all apps were updated with relevant information by the dataset owner.

Let's look at this distribution grouped by category:
""")

with st.echo():
    fig, ax = plt.subplots()

    sns.boxplot(x='category', y='days_elapsed', data=top,
                hue='free', palette=['#74a09e', '#f2a553'])

    ax.set(title='Comparison of Categories By Days Elapsed Since Last Update',
           xlabel='Days Elapsed', ylabel='')

    ax.legend(loc='upper right', title='')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    plt.show()

st.pyplot(fig)

st.markdown("""
The above plot shows a clear trend. Surprisingly, paid apps are much less maintained than free apps except for Food & Drink category. 
""")

st.markdown("""
# 6. Brief Conclusion of the EDA

In short, this dataset clearly showed how competetive the Google Play Store is. More than half of the apps had close to 0 ratings and installs. 

Even though there were more than 50 categories, only top 8 were explored which accounted to more than half of all apps. Among the top 8, `Tools` category was a clear winner in terms of install and rating count. 

Majority of the paid apps cost around a dollar however there were outliers as huge as 400 dollars. 
Coming to paid apps, most expensive categories were Food & Drink and Business. 

In 4th and 5th parts of the exploration, the dataset was looked at from different angles. Specifically, the relationships between rating and rating count, rating and install count, rating and maintenance were explored. 

For better insight, apps with no ratings and installs were exlcuded. This showed that most apps had a rating of ~4.5 and most highly-rated apps were in Music.
""")
