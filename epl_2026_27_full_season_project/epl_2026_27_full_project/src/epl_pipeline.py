from collections import defaultdict, deque
import numpy as np
import pandas as pd

LABEL_ORDER = ["A", "D", "H"]
LABEL_TO_INT = {"A":0, "D":1, "H":2}
CAT_FEATURES = ["HomeTeam", "AwayTeam"]
PREMATCH_FEATURES = ['Home_Elo', 'Away_Elo', 'Elo_Difference', 'Home_PPG_Last5', 'Away_PPG_Last5', 'PPG_Difference', 'Home_WinRate_Last5', 'Away_WinRate_Last5', 'WinRate_Difference', 'Home_GoalsFor_Last5', 'Away_GoalsFor_Last5', 'GoalsFor_Difference', 'Home_GoalsAgainst_Last5', 'Away_GoalsAgainst_Last5', 'GoalsAgainst_Difference', 'Home_GoalDiff_Last5', 'Away_GoalDiff_Last5', 'GoalDiff_Difference', 'Home_HomePPG_Last5', 'Away_AwayPPG_Last5', 'Venue_PPG_Difference', 'Home_RestDays', 'Away_RestDays', 'Rest_Difference', 'H2H_HomeWinRate_Last5', 'H2H_DrawRate_Last5', 'H2H_GoalDiff_Last5', 'Month', 'DayOfWeek']
LIVE_FEATURES = ['HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HT_GoalDifference', 'HomeShots', 'AwayShots', 'Shots_Difference', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'SOT_Difference', 'HomeCorners', 'AwayCorners', 'Corners_Difference', 'HomeFouls', 'AwayFouls', 'Fouls_Difference', 'HomeYellowCards', 'AwayYellowCards', 'Yellow_Difference', 'HomeRedCards', 'AwayRedCards', 'Red_Difference', 'xG_Home', 'xG_Away', 'xG_Difference', 'Home_Shot_Accuracy', 'Away_Shot_Accuracy', 'Shot_Accuracy_Difference', 'Total_Shots', 'Total_SOT', 'Total_Corners', 'Total_Fouls', 'Total_Cards']

# Reusable helpers are documented in the notebook. The production repository
# stores the fitted models in models/ and prediction outputs in results/.

def result_from_goal_difference(gd, threshold=0.25):
    return np.where(gd > threshold, "H", np.where(gd < -threshold, "A", "D"))
