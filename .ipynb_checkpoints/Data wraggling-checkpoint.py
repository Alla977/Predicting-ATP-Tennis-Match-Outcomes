import pandas as pd

# Read the CSV files for 2020-2023
atp_matches_2019 = pd.read_csv("Data/atp_matches_2019.csv", sep=",")
atp_matches_2020 = pd.read_csv("Data/atp_matches_2020.csv", sep=",")
atp_matches_2021 = pd.read_csv("Data/atp_matches_2021.csv", sep=",")
atp_matches_2022 = pd.read_csv("Data/atp_matches_2022.csv", sep=",")
atp_matches_2023 = pd.read_csv("Data/atp_matches_2023.csv", sep=",")

# Concatenate the dataframes vertically (similar to rbind in R)
atp_matches = pd.concat([atp_matches_2019,atp_matches_2020,atp_matches_2021, atp_matches_2022, atp_matches_2023], ignore_index=True)

# Convert the 'surface' column to a categorical variable
atp_matches['surface'] = atp_matches['surface'].astype('category')

# Print the levels of the 'surface' categorical variable (equivalent to levels in R)
print(atp_matches['surface'].cat.categories)

# Main player definition and missing value imputation
# Constructing win indicator
import numpy as np

np.random.seed(42)
atp_matches['Main_player'] = np.where(np.random.binomial(n=1, p=0.5, size=len(atp_matches)) == 1, 'winner', 'loser')
atp_matches['Outcome'] = np.where(atp_matches['Main_player'] == 'winner', 1, 0)
atp_matches['Main_player_name'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['winner_name'],
                                           atp_matches['loser_name'])

# Impute missing values with median
atp_matches['loser_ht'].fillna(atp_matches['loser_ht'].median(), inplace=True)
atp_matches['winner_ht'].fillna(atp_matches['winner_ht'].median(), inplace=True)
atp_matches['loser_age'].fillna(atp_matches['loser_age'].median(), inplace=True)
atp_matches['winner_age'].fillna(atp_matches['winner_age'].median(), inplace=True)
atp_matches['loser_rank'].fillna(atp_matches['loser_rank'].median(), inplace=True)
atp_matches['winner_rank'].fillna(atp_matches['winner_rank'].median(), inplace=True)
atp_matches['l_df'].fillna(atp_matches['l_df'].median(), inplace=True)
atp_matches['w_df'].fillna(atp_matches['w_df'].median(), inplace=True)

# Calculate age_diff, rank_diff, and height_diff
# Ungroup the DataFrame (if needed)
atp_matches = atp_matches.reset_index(drop=True)

# Calculate age_diff, rank_diff, and height_diff
atp_matches['age_diff'] = atp_matches.apply(
    lambda row: row['winner_age'] - row['loser_age'] if row['Main_player'] == 'winner' else row['loser_age'] - row[
        'winner_age'], axis=1)

atp_matches['rank_diff'] = atp_matches.apply(
    lambda row: row['winner_rank'] - row['loser_rank'] if row['Main_player'] == 'winner' else row['loser_rank'] - row[
        'winner_rank'], axis=1)

atp_matches['height_diff'] = atp_matches.apply(
    lambda row: row['winner_ht'] - row['loser_ht'] if row['Main_player'] == 'winner' else row['loser_ht'] - row[
        'winner_ht'], axis=1)

# Calculate average of the latest 10 First Serve Percentages (svIn_Main)

# Arrange by decreasing tournament date
# Define the round mapping dictionary
round_mapping = {
    'R128': 1,
    'RR': 1,
    'BR': 1,
    'R64': 2,
    'R32': 3,
    'R16': 4,
    'QF': 5,
    'SF': 6,
    'F': 7
}

# Convert the 'round' column to integers based on the mapping
atp_matches['round'] = atp_matches['round'].map(round_mapping)

# Handle unmapped values by setting them to a default value (e.g., 0)
atp_matches['round'] = atp_matches['round'].fillna(0)

atp_matches = atp_matches.sort_values(by=['Main_player_name', 'tourney_date','round'], ascending=[True, False,True])

# Define svpt and svIn based on Main_player
atp_matches['svpt_Main'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['w_svpt'],
                                    atp_matches['l_svpt'])
atp_matches['svIn_Main'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['w_1stIn'],
                                    atp_matches['l_1stIn'])

# Define svpt and svIn based on Other_player
atp_matches['svpt_Other'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['l_svpt'],
                                    atp_matches['w_svpt'])
atp_matches['svIn_Other'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['l_1stIn'],
                                    atp_matches['w_1stIn'])

window_size = 10

# Calculate historical first service percent based on both players
atp_matches['h_SVP_Main'] = (atp_matches.groupby('Main_player_name')['svIn_Main'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
    level=0, drop=True) /
                             atp_matches.groupby('Main_player_name')['svpt_Main'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
                                 level=0, drop=True))

atp_matches['h_SVP_Other'] = (atp_matches.groupby('Main_player_name')['svIn_Other'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
    level=0, drop=True) /
                             atp_matches.groupby('Main_player_name')['svpt_Other'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
                                 level=0, drop=True))

atp_matches['h_SVP_diff'] = atp_matches['h_SVP_Main']-atp_matches['h_SVP_Other']

# Calculate historical break points saved percent based on Main_player
# Arrange by decreasing tournament date


atp_matches['bpFaced_Main'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['w_bpFaced'],
                                    atp_matches['l_bpFaced'])
atp_matches['bpSaved_Main'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['w_bpSaved'],
                                    atp_matches['l_bpSaved'])

# Define bpFaced and bpSaved based on Other_player
atp_matches['bpFaced_Other'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['l_bpFaced'],
                                    atp_matches['w_bpFaced'])
atp_matches['bpSaved_Other'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['l_bpSaved'],
                                    atp_matches['w_bpSaved'])

window_size = 10

# Calculate historical break points saved percent based on both players and calculate difference
atp_matches['h_BPSP_Main_10'] = (atp_matches.groupby('Main_player_name')['bpSaved_Main'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
    level=0, drop=True) /
                             atp_matches.groupby('Main_player_name')['bpFaced_Main'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
                                 level=0, drop=True))

atp_matches['h_BPSP_Other_10'] = (atp_matches.groupby('Main_player_name')['bpSaved_Other'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
    level=0, drop=True) /
                             atp_matches.groupby('Main_player_name')['bpFaced_Other'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
                                 level=0, drop=True))

atp_matches['h_BPSP_diff_10'] = atp_matches['h_BPSP_Main_10']-atp_matches['h_BPSP_Other_10']

window_size = 5

# Calculate historical break points saved percent based on both players and calculate difference
atp_matches['h_BPSP_Main_5'] = (atp_matches.groupby('Main_player_name')['bpSaved_Main'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
    level=0, drop=True) /
                             atp_matches.groupby('Main_player_name')['bpFaced_Main'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
                                 level=0, drop=True))

atp_matches['h_BPSP_Other_5'] = (atp_matches.groupby('Main_player_name')['bpSaved_Other'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
    level=0, drop=True) /
                             atp_matches.groupby('Main_player_name')['bpFaced_Other'].rolling(window=window_size,
                                                                                          min_periods=1).sum().reset_index(
                                 level=0, drop=True))

atp_matches['h_BPSP_diff_5'] = atp_matches['h_BPSP_Main_5']-atp_matches['h_BPSP_Other_5']




## Calculating short-term win/loss ratio as a variable
#Steps
#1. Group by date/round and player based on original data and find all matches player partakes in
#2. Calculate WLR
#3. Merge onto atp_matches respectively on main_player and other_player
#4. Calculate the difference
window_size=5
atp_matches['h_WLR'] = (atp_matches.groupby('Main_player_name')['Outcome']
                      .apply(lambda x: x.shift().rolling(window=5, min_periods=1).sum()/ x.index.to_series().rolling(window=5, min_periods=1).count())
                      .reset_index(level=0, drop=True))




# Grouping variables

# Calculate quantiles for age_diff
age_diff_quantiles = np.percentile(atp_matches['age_diff'].dropna(), np.arange(0, 101, 20))

# Calculate quantiles for rank_diff
rank_diff_quantiles = np.percentile(atp_matches['rank_diff'].dropna(), np.arange(0, 101, 20))

# Calculate quantiles for height_diff
height_diff_quantiles = np.percentile(atp_matches['height_diff'].dropna(), np.arange(0, 101, 20))

# Calculate quantiles for h_SVP_Main
h_SVP_quantiles = np.percentile(atp_matches['h_SVP_diff'].dropna(), np.arange(0, 101, 20))

# Create labels for grouping based on quantiles
age_diff_labels = ["[-,-10)", "[-10,-4)", "[-4,-2)", "[-2,0)", "[0,3)", "[3,10)", "[10,+)"]
height_diff_labels = ["[-,-10)", "[-10,-5)", "[-5,0)", "[0,5)", "[5,10)", "[10,+)"]
rank_diff_labels = ["[-,-60)", "[-60,-20)", "[-20,-10)", "[-10,0)", "[0,10)", "[10,20)", "[20,60)", "[60,+)"]
h_SVP_labels = ["[-,-0.1)", "[-0.1,-0.05)", "[-0.05,0)", "[0,0.05)","[0.05,0.1)","[0.1,+)"]
h_BPSP_labels = ["[-,-0.1)", "[-0.1,-0.05)", "[-0.05,0)", "[0,0.05)","[0.05,0.1)","[0.1,+)"]

# Create new columns with labels
atp_matches['age_diff_group'] = pd.cut(atp_matches['age_diff'], bins=[-np.inf, -10, -4, -2, 0, 3, 10, np.inf],
                                       labels=age_diff_labels, right=False)
atp_matches['height_diff_group'] = pd.cut(atp_matches['height_diff'], bins=[-np.inf, -10, -5, 0, 5, 10, np.inf],
                                          labels=height_diff_labels, right=False)
atp_matches['rank_diff_group'] = pd.cut(atp_matches['rank_diff'], bins=[-np.inf, -60, -20, -10, 0, 10, 20, 60, np.inf],
                                        labels=rank_diff_labels, right=False)
atp_matches['h_SVP_group'] = pd.cut(atp_matches['h_SVP_diff'], bins=[-np.inf, -0.1,-0.05, 0, 0.05,0.1,np.inf],
                                    labels=h_SVP_labels, right=False)
atp_matches['h_BPSP_10_group'] = pd.cut(atp_matches['h_BPSP_diff_10'], bins=[-np.inf, -0.1,-0.05, 0, 0.05,0.1,np.inf],
                                    labels=h_BPSP_labels, right=False)
atp_matches['h_BPSP_5_group'] = pd.cut(atp_matches['h_BPSP_diff_5'], bins=[-np.inf, -0.1,-0.05, 0, 0.05,0.1,np.inf],
                                    labels=h_BPSP_labels, right=False)


# Impute missing values with median or mode
atp_matches['h_SVP_diff'].fillna(atp_matches['h_SVP_diff'].median(), inplace=True)
atp_matches['h_BPSP_diff_10'].fillna(atp_matches['h_BPSP_diff_10'].median(), inplace=True)
atp_matches['h_BPSP_diff_5'].fillna(atp_matches['h_BPSP_diff_5'].median(), inplace=True)
atp_matches['surface'].fillna(atp_matches['surface'].mode().iloc[0], inplace=True)
atp_matches['tourney_level'].fillna(atp_matches['tourney_level'].mode().iloc[0], inplace=True)
atp_matches['draw_size'].fillna(atp_matches['draw_size'].mode().iloc[0], inplace=True)


## Including main_player_labels to ease use in 'Data for predictions' file

atp_matches['Main_player_age'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['winner_age'],
                                    atp_matches['loser_age'])
atp_matches['Main_player_height'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['winner_ht'],
                                    atp_matches['loser_ht'])
atp_matches['Main_player_rank'] = np.where(atp_matches['Main_player'] == 'winner', atp_matches['winner_rank'],
                                    atp_matches['loser_rank'])



#Saving data for further use
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(atp_matches[['Main_player_name','surface','tourney_level','draw_size','age_diff','height_diff','rank_diff','h_SVP_diff','h_BPSP_diff_10','h_BPSP_diff_5','Home_crowd','Country']], atp_matches['Outcome'], test_size=0.2, random_state=42)

atp_matches.to_csv('Temp data/atp_matches.csv', index=False)
X_train.to_csv('Temp data/X_train.csv', index=False)  # Set index=False to exclude the index column
X_test.to_csv('Temp data/X_test.csv', index=False)  # Set index=False to exclude the index column
y_train.to_csv('Temp data/y_train.csv', index=False)  # Set index=False to exclude the index column
y_test.to_csv('Temp data/y_test.csv', index=False)  # Set index=False to exclude the index column

