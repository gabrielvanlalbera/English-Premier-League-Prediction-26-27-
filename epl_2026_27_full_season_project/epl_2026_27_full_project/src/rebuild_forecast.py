from pathlib import Path
import json, math, re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, classification_report
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parents[1]
for p in ['data/raw','data/processed','models','results']:
    (BASE/p).mkdir(parents=True, exist_ok=True)

# -------------------- team normalization --------------------
ALIASES = {
    'Nott\'ham Forest': "Nott'm Forest", 'Nottingham Forest': "Nott'm Forest", 'Nottingham Forest FC': "Nott'm Forest",
    'Manchester United': 'Man United', 'Manchester Utd': 'Man United', 'Man Utd': 'Man United', 'Manchester United FC': 'Man United',
    'Manchester City': 'Man City', 'Manchester City FC': 'Man City', 'Man City': 'Man City',
    'Tottenham Hotspur': 'Tottenham', 'Tottenham Hotspur FC': 'Tottenham', 'Spurs': 'Tottenham', 'Tottenham FC': 'Tottenham',
    'AFC Bournemouth': 'Bournemouth', 'Bournemouth FC': 'Bournemouth', 'Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton', 'Brighton & Hove Albion FC': 'Brighton', 'Brighton': 'Brighton',
    'Leeds United': 'Leeds', 'Leeds United FC': 'Leeds', 'Leeds': 'Leeds',
    'Ipswich Town': 'Ipswich', 'Ipswich Town FC': 'Ipswich', 'Ipswich': 'Ipswich',
    'Hull City': 'Hull', 'Hull City AFC': 'Hull', 'Hull': 'Hull',
    'Coventry City': 'Coventry', 'Coventry City FC': 'Coventry', 'Coventry': 'Coventry',
    'Newcastle United': 'Newcastle', 'Newcastle United FC': 'Newcastle', 'Newcastle Utd': 'Newcastle', 'Newcastle': 'Newcastle',
    'Crystal Palace FC': 'Crystal Palace',
    'Aston Villa FC': 'Aston Villa',
    'Everton FC': 'Everton', 'Arsenal FC': 'Arsenal', 'Chelsea FC': 'Chelsea', 'Liverpool FC': 'Liverpool',
    'Fulham FC': 'Fulham', 'Sunderland AFC': 'Sunderland', 'Sunderland': 'Sunderland',
    'Brentford FC': 'Brentford',
    'Burnley FC': 'Burnley', 'West Ham United FC': 'West Ham', 'West Ham Utd': 'West Ham',
    'Wolves FC': 'Wolves', 'Wolverhampton Wanderers': 'Wolves',
    'Leicester City': 'Leicester', 'Leicester City FC': 'Leicester',
    'Sheffield Utd': 'Sheffield United', 'Sheffield United FC': 'Sheffield United',
    'Luton Town': 'Luton', 'Luton Town FC': 'Luton',
    'Southampton FC': 'Southampton',
    'Birmingham City': 'Birmingham',
    'West Bromwich Albion': 'West Brom',
}

def norm_team(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return ALIASES.get(s, s)

# -------------------- load/merge data --------------------
orig = pd.read_csv(BASE/'data/raw/epl_final.csv')
orig['MatchDate'] = pd.to_datetime(orig['MatchDate'], errors='coerce')
for c in ['HomeTeam','AwayTeam']:
    orig[c] = orig[c].map(norm_team)
orig['FullTimeResult'] = orig['FullTimeResult'].replace({'H':'H','D':'D','A':'A'})

stats = pd.read_csv(BASE/'data/raw/epl-allseasons-matchstats.csv')
stats['Date'] = pd.to_datetime(stats['Date'], format='%d-%m-%Y', errors='coerce')
stats['HomeTeam'] = stats['HomeTeam'].map(norm_team)
stats['AwayTeam'] = stats['AwayTeam'].map(norm_team)
stats = stats.rename(columns={'Date':'MatchDate'})
stats['HomeShots'] = pd.to_numeric(stats['HomeShots'], errors='coerce')
stats['AwayShots'] = pd.to_numeric(stats['AwayShots'], errors='coerce')
stats['HomeShotsOnTarget'] = pd.to_numeric(stats['HomeShotsOnTarget'], errors='coerce')
stats['AwayShotsOnTarget'] = pd.to_numeric(stats['AwayShotsOnTarget'], errors='coerce')

book = pd.read_excel(BASE/'data/raw/Book1.xlsx', sheet_name='match_data')
book['Date'] = pd.to_datetime(book['Date'], errors='coerce')
book['HomeTeam'] = book['Home'].map(norm_team)
book['AwayTeam'] = book['Away'].map(norm_team)
book['Season'] = book['Season'].astype(str).str.replace('-', '/', regex=False).str.replace(r'/(\d{4})$', lambda m: '/' + m.group(1)[-2:], regex=True)
book['xG_Home'] = pd.to_numeric(book['xG'], errors='coerce')
book['xG_Away'] = pd.to_numeric(book['xG.1'], errors='coerce')
book['FullTimeHomeGoals'] = pd.to_numeric(book['FullTimeHomeGoals'], errors='coerce')
book['FullTimeAwayGoals'] = pd.to_numeric(book['FullTimeAwayGoals'], errors='coerce')
book['FullTimeResult'] = pd.Series(np.where(book['FullTimeHomeGoals'] > book['FullTimeAwayGoals'], 'H', np.where(book['FullTimeHomeGoals'] < book['FullTimeAwayGoals'], 'A', 'D')), index=book.index, dtype='object')
book.loc[book['FullTimeHomeGoals'].isna() | book['FullTimeAwayGoals'].isna(), 'FullTimeResult'] = np.nan
book = book[['Season','Date','HomeTeam','AwayTeam','Referee','xG_Home','xG_Away','FullTimeHomeGoals','FullTimeAwayGoals','FullTimeResult']].rename(columns={'Date':'MatchDate'})

# Create match keys; original dataset is authoritative for core match stats.
def key(df):
    return (df['MatchDate'].dt.strftime('%Y-%m-%d')+'|'+df['HomeTeam']+'|'+df['AwayTeam'])
orig['MatchKey'] = key(orig)
stats['MatchKey'] = key(stats)
book['MatchKey'] = key(book)

# Enrich original with referee/xG from the additional sources.
ref_map_stats = stats.dropna(subset=['Referee']).drop_duplicates('MatchKey').set_index('MatchKey')['Referee']
ref_map_book = book.dropna(subset=['Referee']).drop_duplicates('MatchKey').set_index('MatchKey')['Referee']
xgh_map = book.drop_duplicates('MatchKey').set_index('MatchKey')['xG_Home']
xga_map = book.drop_duplicates('MatchKey').set_index('MatchKey')['xG_Away']
orig['Referee'] = orig['MatchKey'].map(ref_map_stats).fillna(orig['MatchKey'].map(ref_map_book))
orig['xG_Home'] = orig['MatchKey'].map(xgh_map)
orig['xG_Away'] = orig['MatchKey'].map(xga_map)

# Add missing completed records from Book1 for 2024/25 and 2025/26.
book_add = book[book['FullTimeResult'].notna() & book['MatchKey'].notna()].copy()
existing = set(orig['MatchKey'])
new_additions = book_add[~book_add['MatchKey'].isin(existing)].copy()
# Book1 does not contain the full live-stat columns used by the original source.
for c in ['HalfTimeHomeGoals','HalfTimeAwayGoals','HalfTimeResult','HomeShots','AwayShots','HomeShotsOnTarget','AwayShotsOnTarget','HomeCorners','AwayCorners','HomeFouls','AwayFouls','HomeYellowCards','AwayYellowCards','HomeRedCards','AwayRedCards']:
    new_additions[c] = np.nan
new_additions = new_additions[orig.columns]

combined = pd.concat([orig, new_additions], ignore_index=True)
combined = combined.drop_duplicates('MatchKey').sort_values('MatchDate').reset_index(drop=True)
combined['FullTimeHomeGoals'] = pd.to_numeric(combined['FullTimeHomeGoals'], errors='coerce')
combined['FullTimeAwayGoals'] = pd.to_numeric(combined['FullTimeAwayGoals'], errors='coerce')
combined['GoalDifference'] = combined['FullTimeHomeGoals'] - combined['FullTimeAwayGoals']
combined.to_csv(BASE/'data/processed/epl_combined_historical.csv', index=False)

# -------------------- referee profile --------------------
ref = combined[combined['FullTimeResult'].notna() & combined['Referee'].notna()].copy()
for c in ['HomeFouls','AwayFouls','HomeYellowCards','AwayYellowCards','HomeRedCards','AwayRedCards']:
    ref[c] = pd.to_numeric(ref[c], errors='coerce')
ref['TotalFouls'] = ref['HomeFouls'] + ref['AwayFouls']
ref['TotalYellow'] = ref['HomeYellowCards'] + ref['AwayYellowCards']
ref['TotalRed'] = ref['HomeRedCards'] + ref['AwayRedCards']
ref['HomeYellowBias'] = ref['HomeYellowCards'] - ref['AwayYellowCards']
profile = ref.groupby('Referee').agg(
    Matches=('FullTimeResult','size'),
    Avg_Total_Fouls=('TotalFouls','mean'),
    Avg_Total_Yellow=('TotalYellow','mean'),
    Avg_Total_Red=('TotalRed','mean'),
    Home_Yellow_Bias=('HomeYellowBias','mean'),
    Home_Win_Rate=('FullTimeResult',lambda s:(s=='H').mean()),
    Draw_Rate=('FullTimeResult',lambda s:(s=='D').mean()),
    Away_Win_Rate=('FullTimeResult',lambda s:(s=='A').mean()),
).reset_index().sort_values(['Matches','Avg_Total_Yellow'], ascending=[False,False])
profile.to_csv(BASE/'data/processed/referee_impact_profile.csv', index=False)
profile[profile['Matches']>=20].to_csv(BASE/'results/referee_profiles_20plus_matches.csv', index=False)

# -------------------- Current 2026/27 results (completed MW1-5) --------------------
cur_text = '''1|Arsenal|Coventry|3|0
1|Hull|Man United|2|0
1|Ipswich|Sunderland|2|1
1|Nott'm Forest|Leeds|0|1
1|Everton|Crystal Palace|2|0
1|Brentford|Tottenham|3|0
1|Man City|Bournemouth|2|1
1|Brighton|Aston Villa|4|0
1|Newcastle|Liverpool|2|2
1|Fulham|Chelsea|2|3
2|Crystal Palace|Man City|1|4
2|Liverpool|Nott'm Forest|2|2
2|Bournemouth|Everton|1|1
2|Coventry|Hull|0|1
2|Tottenham|Newcastle|0|2
2|Sunderland|Fulham|1|0
2|Chelsea|Brighton|4|3
2|Leeds|Brentford|1|1
2|Man United|Ipswich|5|2
2|Aston Villa|Arsenal|0|1
3|Ipswich|Liverpool|0|2
3|Newcastle|Bournemouth|2|2
3|Brentford|Sunderland|1|1
3|Brighton|Leeds|1|1
3|Fulham|Crystal Palace|2|3
3|Man City|Coventry|1|0
3|Nott'm Forest|Tottenham|0|0
3|Hull|Aston Villa|0|0
3|Everton|Man United|2|2
3|Arsenal|Chelsea|2|1
4|Bournemouth|Brentford|2|2
4|Aston Villa|Nott'm Forest|1|2
4|Chelsea|Hull|2|2
4|Crystal Palace|Ipswich|2|3
4|Liverpool|Fulham|0|0
4|Tottenham|Everton|0|0
4|Sunderland|Arsenal|0|2
4|Coventry|Brighton|0|5
4|Man United|Man City|0|1
4|Leeds|Newcastle|4|1
5|Brentford|Chelsea|3|0
5|Tottenham|Aston Villa|2|3
5|Everton|Ipswich|1|0
5|Brighton|Arsenal|3|0
5|Newcastle|Hull|2|1
5|Nott'm Forest|Coventry|0|1
5|Bournemouth|Liverpool|0|1
5|Leeds|Crystal Palace|0|0
5|Man City|Sunderland|5|3
5|Fulham|Man United|1|1'''
rows=[]
for line in cur_text.strip().splitlines():
    mw,h,a,hg,ag=line.split('|')
    rows.append({'Matchweek':int(mw),'MatchDate':pd.NaT,'HomeTeam':h,'AwayTeam':a,'FullTimeHomeGoals':int(hg),'FullTimeAwayGoals':int(ag)})
current_results = pd.DataFrame(rows)
current_results['FullTimeResult']=np.where(current_results.FullTimeHomeGoals>current_results.FullTimeAwayGoals,'H',np.where(current_results.FullTimeHomeGoals<current_results.FullTimeAwayGoals,'A','D'))
current_results['Season']='2026/27'
current_results['MatchDate'] = pd.to_datetime(['2026-08-21']*10+['2026-08-28']*10+['2026-09-04']*10+['2026-09-12']*10+['2026-09-18']*10)  # anchor only for state ordering
current_results['MatchKey']=[f'2026-27-{i+1}' for i in range(len(current_results))]
current_results.to_csv(BASE/'data/processed/2026_27_completed_mw1_mw5.csv', index=False)

# For state-building, use an actual sequence order rather than equal dates.
current_results['_order'] = np.arange(len(current_results))

# -------------------- Historical state/feature builder --------------------
BASE_FEATURES = [
    'Home_Elo','Away_Elo','Elo_Difference',
    'Home_PPG_Last5','Away_PPG_Last5','PPG_Difference',
    'Home_WinRate_Last5','Away_WinRate_Last5','WinRate_Difference',
    'Home_GoalsFor_Last5','Away_GoalsFor_Last5','GoalsFor_Difference',
    'Home_GoalsAgainst_Last5','Away_GoalsAgainst_Last5','GoalsAgainst_Difference',
    'Home_GoalDiff_Last5','Away_GoalDiff_Last5','GoalDiff_Difference',
    'Home_HomePPG_Last5','Away_AwayPPG_Last5','Venue_PPG_Difference',
    'Home_RestDays','Away_RestDays','Rest_Difference',
    'H2H_HomeWinRate_Last5','H2H_DrawRate_Last5','H2H_GoalDiff_Last5',
    'Month','DayOfWeek'
]
CAT_FEATURES=['HomeTeam','AwayTeam']
LIVE_FEATURES=[
    'HalfTimeHomeGoals','HalfTimeAwayGoals','HT_GoalDifference',
    'HomeShots','AwayShots','Shots_Difference',
    'HomeShotsOnTarget','AwayShotsOnTarget','SOT_Difference',
    'HomeCorners','AwayCorners','Corners_Difference',
    'HomeFouls','AwayFouls','Fouls_Difference',
    'HomeYellowCards','AwayYellowCards','Yellow_Difference',
    'HomeRedCards','AwayRedCards','Red_Difference',
    'xG_Home','xG_Away','xG_Difference',
    'Home_Shot_Accuracy','Away_Shot_Accuracy','Shot_Accuracy_Difference',
    'Total_Shots','Total_SOT','Total_Corners','Total_Fouls','Total_Cards',
]


def hist_vals(hist, key):
    return [x[key] for x in hist]

def mean_or_zero(vals): return float(np.mean(vals)) if vals else 0.0

def ppg_last(hist): return mean_or_zero(hist_vals(hist,'points'))

def winrate_last(hist): return mean_or_zero(hist_vals(hist,'win'))

def build_state_features(data):
    data=data.sort_values('MatchDate').copy()
    # when dates are equal, retain original order
    data['_seq']=np.arange(len(data))
    data=data.sort_values(['MatchDate','_seq'], kind='stable')
    states=defaultdict(lambda: {
        'elo':1500.0,'hist':deque(maxlen=5),'home_hist':deque(maxlen=5),'away_hist':deque(maxlen=5),
        'last_date':None
    })
    h2h=defaultdict(lambda: deque(maxlen=5))
    out=[]
    for _,r in data.iterrows():
        h,a=r['HomeTeam'],r['AwayTeam']
        hs,as_ = states[h],states[a]
        hh=list(hs['hist']); ah=list(as_['hist'])
        hhh=list(hs['home_hist']); aah=list(as_['away_hist'])
        keypair=(h,a)
        pair_hist=list(h2h[keypair])
        if pair_hist:
            h2h_wr=mean_or_zero([x['home_win'] for x in pair_hist])
            h2h_dr=mean_or_zero([x['draw'] for x in pair_hist])
            h2h_gd=mean_or_zero([x['gd'] for x in pair_hist])
        else:
            h2h_wr=h2h_dr=h2h_gd=0.0
        dt=r['MatchDate']
        hrest=(dt-hs['last_date']).days if hs['last_date'] is not None else 14
        arest=(dt-as_['last_date']).days if as_['last_date'] is not None else 14
        row={
            'MatchDate':dt,'Season':r.get('Season',None),'HomeTeam':h,'AwayTeam':a,
            'Home_Elo':hs['elo'],'Away_Elo':as_['elo'],'Elo_Difference':hs['elo']+50-as_['elo'],
            'Home_PPG_Last5':ppg_last(hh),'Away_PPG_Last5':ppg_last(ah),'PPG_Difference':ppg_last(hh)-ppg_last(ah),
            'Home_WinRate_Last5':winrate_last(hh),'Away_WinRate_Last5':winrate_last(ah),'WinRate_Difference':winrate_last(hh)-winrate_last(ah),
            'Home_GoalsFor_Last5':mean_or_zero(hist_vals(hh,'gf')),'Away_GoalsFor_Last5':mean_or_zero(hist_vals(ah,'gf')),
            'GoalsFor_Difference':mean_or_zero(hist_vals(hh,'gf'))-mean_or_zero(hist_vals(ah,'gf')),
            'Home_GoalsAgainst_Last5':mean_or_zero(hist_vals(hh,'ga')),'Away_GoalsAgainst_Last5':mean_or_zero(hist_vals(ah,'ga')),
            'GoalsAgainst_Difference':mean_or_zero(hist_vals(hh,'ga'))-mean_or_zero(hist_vals(ah,'ga')),
            'Home_GoalDiff_Last5':mean_or_zero([x['gf']-x['ga'] for x in hh]),'Away_GoalDiff_Last5':mean_or_zero([x['gf']-x['ga'] for x in ah]),
            'GoalDiff_Difference':mean_or_zero([x['gf']-x['ga'] for x in hh])-mean_or_zero([x['gf']-x['ga'] for x in ah]),
            'Home_HomePPG_Last5':ppg_last(hhh),'Away_AwayPPG_Last5':ppg_last(aah),'Venue_PPG_Difference':ppg_last(hhh)-ppg_last(aah),
            'Home_RestDays':hrest,'Away_RestDays':arest,'Rest_Difference':hrest-arest,
            'H2H_HomeWinRate_Last5':h2h_wr,'H2H_DrawRate_Last5':h2h_dr,'H2H_GoalDiff_Last5':h2h_gd,
            'Month':dt.month if pd.notna(dt) else 1,'DayOfWeek':dt.dayofweek if pd.notna(dt) else 5
        }
        out.append(row)
        # Update only if result available
        if pd.isna(r.get('FullTimeHomeGoals')) or pd.isna(r.get('FullTimeAwayGoals')):
            continue
        hg,ag=float(r['FullTimeHomeGoals']),float(r['FullTimeAwayGoals'])
        if hg>ag: hpoints,apoints,hwin,awin=3,0,1,0; actual=1
        elif hg<ag: hpoints,apoints,hwin,awin=0,3,0,1; actual=0
        else: hpoints,apoints,hwin,awin=1,1,0,0; actual=0.5
        exp=1/(1+10**(((as_['elo'])-(hs['elo']+50))/400))
        delta=20*(actual-exp)
        hs['elo'] += delta; as_['elo'] -= delta
        hs['hist'].append({'points':hpoints,'win':hwin,'gf':hg,'ga':ag})
        as_['hist'].append({'points':apoints,'win':awin,'gf':ag,'ga':hg})
        hs['home_hist'].append({'points':hpoints,'win':hwin,'gf':hg,'ga':ag})
        as_['away_hist'].append({'points':apoints,'win':awin,'gf':ag,'ga':hg})
        h2h[keypair].append({'home_win':hwin,'draw':1 if hg==ag else 0,'gd':hg-ag})
        hs['last_date']=dt; as_['last_date']=dt
    feat=pd.DataFrame(out)
    # Return state by extracting internal values after processing.
    state={}
    for t,s in states.items():
        state[t]={
            'elo':s['elo'],'hist':list(s['hist']),'home_hist':list(s['home_hist']),'away_hist':list(s['away_hist']),'last_date':s['last_date']
        }
    h2h_state={k:list(v) for k,v in h2h.items()}
    return feat,state,h2h_state

# Historical features
complete_hist=combined[combined['FullTimeResult'].notna()].copy()
hist_features,state_hist,h2h_hist=build_state_features(complete_hist)
# Add target and match level fields
merged_feat=hist_features.merge(complete_hist[['MatchKey','FullTimeResult','FullTimeHomeGoals','FullTimeAwayGoals','GoalDifference','Referee','xG_Home','xG_Away']], left_index=True,right_index=True,how='left')
# current stats features for optional in-match model
cf=complete_hist.reset_index(drop=True)
current_live=pd.DataFrame(index=cf.index)
for c in LIVE_FEATURES:
    current_live[c]=np.nan
mapcols={'HalfTimeHomeGoals':'HalfTimeHomeGoals','HalfTimeAwayGoals':'HalfTimeAwayGoals','HomeShots':'HomeShots','AwayShots':'AwayShots','HomeShotsOnTarget':'HomeShotsOnTarget','AwayShotsOnTarget':'AwayShotsOnTarget','HomeCorners':'HomeCorners','AwayCorners':'AwayCorners','HomeFouls':'HomeFouls','AwayFouls':'AwayFouls','HomeYellowCards':'HomeYellowCards','AwayYellowCards':'AwayYellowCards','HomeRedCards':'HomeRedCards','AwayRedCards':'AwayRedCards','xG_Home':'xG_Home','xG_Away':'xG_Away'}
for outc,inc in mapcols.items():
    if inc in cf.columns:
        current_live[outc]=pd.to_numeric(cf[inc],errors='coerce')
current_live['HT_GoalDifference']=current_live['HalfTimeHomeGoals']-current_live['HalfTimeAwayGoals']
current_live['Shots_Difference']=current_live['HomeShots']-current_live['AwayShots']
current_live['SOT_Difference']=current_live['HomeShotsOnTarget']-current_live['AwayShotsOnTarget']
current_live['Corners_Difference']=current_live['HomeCorners']-current_live['AwayCorners']
current_live['Fouls_Difference']=current_live['HomeFouls']-current_live['AwayFouls']
current_live['Yellow_Difference']=current_live['HomeYellowCards']-current_live['AwayYellowCards']
current_live['Red_Difference']=current_live['HomeRedCards']-current_live['AwayRedCards']
current_live['xG_Difference']=current_live['xG_Home']-current_live['xG_Away']
current_live['Home_Shot_Accuracy']=(current_live['HomeShotsOnTarget']/current_live['HomeShots'].replace(0,np.nan)).fillna(0)
current_live['Away_Shot_Accuracy']=(current_live['AwayShotsOnTarget']/current_live['AwayShots'].replace(0,np.nan)).fillna(0)
current_live['Shot_Accuracy_Difference']=current_live['Home_Shot_Accuracy']-current_live['Away_Shot_Accuracy']
current_live['Total_Shots']=current_live['HomeShots']+current_live['AwayShots']
current_live['Total_SOT']=current_live['HomeShotsOnTarget']+current_live['AwayShotsOnTarget']
current_live['Total_Corners']=current_live['HomeCorners']+current_live['AwayCorners']
current_live['Total_Fouls']=current_live['HomeFouls']+current_live['AwayFouls']
current_live['Total_Cards']=current_live['HomeYellowCards']+current_live['AwayYellowCards']+current_live['HomeRedCards']+current_live['AwayRedCards']
model_table=pd.concat([merged_feat.reset_index(drop=True), current_live.reset_index(drop=True)], axis=1)
model_table = model_table.loc[:, ~model_table.columns.duplicated()].copy()
model_table.to_csv(BASE/'data/processed/engineered_historical_features.csv', index=False)

# -------------------- build model --------------------
pre_features=BASE_FEATURES
train=model_table[model_table['MatchDate'] < pd.Timestamp('2024-08-01')].copy()
test=model_table[(model_table['MatchDate'] >= pd.Timestamp('2024-08-01')) & (model_table['MatchDate'] < pd.Timestamp('2025-08-01'))].copy()
final_train=model_table.copy()

num_features=[c for c in pre_features if c not in CAT_FEATURES]
def make_prep(scale=False):
    num_pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]) if scale else Pipeline([('imp',SimpleImputer(strategy='median'))])
    return ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CAT_FEATURES),('num',num_pipe,num_features)])

def clean(d):
    x=d[CAT_FEATURES + pre_features].copy()
    return x

lr_pre=Pipeline([('prep',make_prep(scale=True)),('model',LinearRegression())])
lr_pre.fit(clean(train),train['GoalDifference'])
lr_gd=lr_pre.predict(clean(test))
lr_pred=np.where(lr_gd>0.25,'H',np.where(lr_gd<-0.25,'A','D'))

prep_xgb=make_prep(scale=False)
Xtr=prep_xgb.fit_transform(clean(train)); Xte=prep_xgb.transform(clean(test))
label_order=['A','D','H']; label_to_int={'A':0,'D':1,'H':2}
xgb=XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',n_estimators=450,max_depth=4,learning_rate=0.04,min_child_weight=4,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.1,reg_lambda=1.8,random_state=42,n_jobs=-1)
xgb.fit(Xtr,train['FullTimeResult'].map(label_to_int),eval_set=[(Xte,test['FullTimeResult'].map(label_to_int))],verbose=False)
xgb_pred_int=xgb.predict(Xte).astype(int); xgb_pred=np.array(label_order)[xgb_pred_int]
xgb_prob=xgb.predict_proba(Xte)

metrics=pd.DataFrame([
    ['Linear Regression — goal difference',accuracy_score(test['FullTimeResult'],lr_pred),f1_score(test['FullTimeResult'],lr_pred,average='macro'),mean_absolute_error(test['GoalDifference'],lr_gd),np.sqrt(mean_squared_error(test['GoalDifference'],lr_gd)),r2_score(test['GoalDifference'],lr_gd)],
    ['XGBoost — H/D/A',accuracy_score(test['FullTimeResult'],xgb_pred),f1_score(test['FullTimeResult'],xgb_pred,average='macro'),np.nan,np.nan,np.nan]
],columns=['Model','Accuracy','Macro_F1','MAE','RMSE','R2'])
metrics.to_csv(BASE/'results/model_metrics_2024_25.csv',index=False)
with open(BASE/'results/classification_report_2024_25.txt','w') as f: f.write(classification_report(test['FullTimeResult'],xgb_pred,labels=label_order,zero_division=0))

# final fit
final_prep=make_prep(False)
Xfull=final_prep.fit_transform(clean(final_train))
final_xgb=XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',n_estimators=450,max_depth=4,learning_rate=0.04,min_child_weight=4,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.1,reg_lambda=1.8,random_state=42,n_jobs=-1)
final_xgb.fit(Xfull,final_train['FullTimeResult'].map(label_to_int),verbose=False)
final_lr=Pipeline([('prep',make_prep(scale=True)),('model',LinearRegression())])
final_lr.fit(clean(final_train),final_train['GoalDifference'])
joblib.dump({'preprocessor':final_prep,'model':final_xgb,'features':pre_features,'label_order':label_order},BASE/'models/prematch_xgboost.joblib')
joblib.dump(final_lr,BASE/'models/prematch_linear_regression.joblib')

# in-match model: use rows with all required base live features, excluding final score
live_req=LIVE_FEATURES.copy()
live_cols=list(dict.fromkeys(CAT_FEATURES+pre_features+live_req))
live_num=[c for c in live_cols if c not in CAT_FEATURES]
live_data=model_table[model_table[live_cols].notna().all(axis=1)].copy()
# Use through 2024/25 for test and all for final
live_train=live_data[live_data['MatchDate']<pd.Timestamp('2024-08-01')]
live_test=live_data[(live_data['MatchDate']>=pd.Timestamp('2024-08-01'))&(live_data['MatchDate']<pd.Timestamp('2025-08-01'))]
def live_prep(scale=False):
    num_pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]) if scale else Pipeline([('imp',SimpleImputer(strategy='median'))])
    return ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CAT_FEATURES),('num',num_pipe,live_num)])
lp=live_prep(False); XLtr=lp.fit_transform(live_train[live_cols]); XLte=lp.transform(live_test[live_cols])
xlgb=XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',n_estimators=400,max_depth=4,learning_rate=0.035,min_child_weight=4,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.1,reg_lambda=2.0,random_state=42,n_jobs=-1)
xlgb.fit(XLtr,live_train['FullTimeResult'].map(label_to_int),verbose=False)
xlgb_pred=np.array(label_order)[xlgb.predict(XLte).astype(int)]
# live linear baseline
llr=Pipeline([('prep',live_prep(True)),('model',LinearRegression())]); llr.fit(live_train[live_cols],live_train['GoalDifference']); llr_gd=llr.predict(live_test[live_cols]); llr_pred=np.where(llr_gd>0.25,'H',np.where(llr_gd<-0.25,'A','D'))
live_metrics=pd.DataFrame([
    ['Linear Regression — in-match',accuracy_score(live_test['FullTimeResult'],llr_pred),f1_score(live_test['FullTimeResult'],llr_pred,average='macro'),mean_absolute_error(live_test['GoalDifference'],llr_gd),np.sqrt(mean_squared_error(live_test['GoalDifference'],llr_gd)),r2_score(live_test['GoalDifference'],llr_gd)],
    ['XGBoost — in-match',accuracy_score(live_test['FullTimeResult'],xlgb_pred),f1_score(live_test['FullTimeResult'],xlgb_pred,average='macro'),np.nan,np.nan,np.nan]
],columns=['Model','Accuracy','Macro_F1','MAE','RMSE','R2'])
live_metrics.to_csv(BASE/'results/model_metrics_inmatch_2024_25.csv',index=False)
# final live
flp=live_prep(False); Xlivefull=flp.fit_transform(live_data[live_cols]); final_live=XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',n_estimators=400,max_depth=4,learning_rate=0.035,min_child_weight=4,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.1,reg_lambda=2.0,random_state=42,n_jobs=-1)
final_live.fit(Xlivefull,live_data['FullTimeResult'].map(label_to_int),verbose=False)
final_live_lr=Pipeline([('prep',live_prep(True)),('model',LinearRegression())]); final_live_lr.fit(live_data[live_cols],live_data['GoalDifference'])
joblib.dump({'preprocessor':flp,'model':final_live,'features':live_cols,'label_order':label_order},BASE/'models/inmatch_xgboost.joblib')
joblib.dump(final_live_lr,BASE/'models/inmatch_linear_regression.joblib')

# feature importance table
imp=pd.Series(final_xgb.feature_importances_,index=final_prep.get_feature_names_out()).sort_values(ascending=False).head(30).reset_index()
imp.columns=['Feature','Importance']; imp.to_csv(BASE/'results/top_30_prematch_xgboost_features.csv',index=False)

# -------------------- Future fixtures --------------------
fixture_text='''6|Arsenal|Leeds
6|Sunderland|Brighton
6|Chelsea|Bournemouth
6|Ipswich|Fulham
6|Aston Villa|Brentford
6|Man United|Tottenham
6|Hull|Everton
6|Crystal Palace|Nott'm Forest
6|Liverpool|Man City
6|Coventry|Newcastle
7|Everton|Chelsea
7|Fulham|Hull
7|Man City|Ipswich
7|Brentford|Liverpool
7|Newcastle|Aston Villa
7|Bournemouth|Sunderland
7|Leeds|Man United
7|Brighton|Crystal Palace
7|Nott'm Forest|Arsenal
7|Tottenham|Coventry
8|Ipswich|Nott'm Forest
8|Aston Villa|Man City
8|Arsenal|Everton
8|Coventry|Fulham
8|Chelsea|Tottenham
8|Liverpool|Brighton
8|Crystal Palace|Newcastle
8|Man United|Bournemouth
8|Hull|Brentford
8|Sunderland|Leeds
9|Chelsea|Man United
9|Man City|Brighton
9|Bournemouth|Leeds
9|Coventry|Sunderland
9|Hull|Ipswich
9|Brentford|Nott'm Forest
9|Tottenham|Crystal Palace
9|Aston Villa|Fulham
9|Liverpool|Arsenal
9|Newcastle|Everton
10|Arsenal|Hull
10|Nott'm Forest|Man City
10|Leeds|Tottenham
10|Fulham|Newcastle
10|Everton|Coventry
10|Crystal Palace|Liverpool
10|Sunderland|Chelsea
10|Ipswich|Bournemouth
10|Brighton|Brentford
10|Man United|Aston Villa
11|Chelsea|Leeds
11|Man City|Fulham
11|Aston Villa|Sunderland
11|Liverpool|Man United
11|Hull|Brighton
11|Bournemouth|Nott'm Forest
11|Tottenham|Ipswich
11|Brentford|Everton
11|Coventry|Crystal Palace
11|Newcastle|Arsenal
12|Leeds|Coventry
12|Ipswich|Aston Villa
12|Arsenal|Man City
12|Man United|Brentford
12|Everton|Liverpool
12|Fulham|Bournemouth
12|Nott'm Forest|Chelsea
12|Brighton|Newcastle
12|Sunderland|Tottenham
12|Crystal Palace|Hull
13|Tottenham|Fulham
13|Liverpool|Sunderland
13|Hull|Nott'm Forest
13|Chelsea|Crystal Palace
13|Aston Villa|Everton
13|Brentford|Arsenal
13|Coventry|Ipswich
13|Man City|Leeds
13|Newcastle|Man United
13|Bournemouth|Brighton
14|Tottenham|Arsenal
14|Man United|Coventry
14|Leeds|Ipswich
14|Newcastle|Sunderland
14|Chelsea|Liverpool
14|Brentford|Man City
14|Bournemouth|Hull
14|Aston Villa|Crystal Palace
14|Nott'm Forest|Brighton
14|Everton|Fulham
15|Sunderland|Nott'm Forest
15|Crystal Palace|Man United
15|Hull|Tottenham
15|Brighton|Everton
15|Ipswich|Newcastle
15|Coventry|Aston Villa
15|Arsenal|Bournemouth
15|Man City|Chelsea
15|Liverpool|Leeds
15|Fulham|Brentford
16|Leeds|Fulham
16|Brighton|Ipswich
16|Nott'm Forest|Everton
16|Brentford|Newcastle
16|Man City|Hull
16|Sunderland|Crystal Palace
16|Bournemouth|Coventry
16|Arsenal|Man United
16|Liverpool|Tottenham
16|Chelsea|Aston Villa
17|Aston Villa|Leeds
17|Everton|Sunderland
17|Fulham|Brighton
17|Tottenham|Bournemouth
17|Ipswich|Brentford
17|Hull|Liverpool
17|Newcastle|Man City
17|Coventry|Chelsea
17|Man United|Nott'm Forest
17|Crystal Palace|Arsenal
18|Hull|Leeds
18|Tottenham|Brighton
18|Everton|Man City
18|Fulham|Arsenal
18|Coventry|Brentford
18|Newcastle|Nott'm Forest
18|Man United|Sunderland
18|Crystal Palace|Bournemouth
18|Aston Villa|Liverpool
18|Ipswich|Chelsea
19|Leeds|Everton
19|Brighton|Man United
19|Brentford|Crystal Palace
19|Arsenal|Ipswich
19|Sunderland|Hull
19|Liverpool|Coventry
19|Nott'm Forest|Fulham
19|Bournemouth|Aston Villa
19|Chelsea|Newcastle
19|Man City|Tottenham
20|Arsenal|Brentford
20|Brighton|Bournemouth
20|Nott'm Forest|Hull
20|Everton|Aston Villa
20|Sunderland|Liverpool
20|Fulham|Tottenham
20|Crystal Palace|Chelsea
20|Ipswich|Coventry
20|Leeds|Man City
20|Man United|Newcastle
21|Bournemouth|Ipswich
21|Man City|Nott'm Forest
21|Chelsea|Sunderland
21|Brentford|Brighton
21|Aston Villa|Man United
21|Tottenham|Leeds
21|Coventry|Everton
21|Liverpool|Crystal Palace
21|Hull|Arsenal
21|Newcastle|Fulham
22|Ipswich|Hull
22|Everton|Brentford
22|Sunderland|Coventry
22|Nott'm Forest|Bournemouth
22|Arsenal|Newcastle
22|Leeds|Chelsea
22|Fulham|Aston Villa
22|Man United|Liverpool
22|Brighton|Man City
22|Crystal Palace|Tottenham
23|Tottenham|Sunderland
23|Brentford|Man United
23|Newcastle|Brighton
23|Hull|Crystal Palace
23|Coventry|Leeds
23|Bournemouth|Fulham
23|Chelsea|Nott'm Forest
23|Liverpool|Everton
23|Man City|Arsenal
23|Aston Villa|Ipswich
24|Arsenal|Liverpool
24|Brighton|Hull
24|Sunderland|Aston Villa
24|Crystal Palace|Coventry
24|Ipswich|Tottenham
24|Everton|Newcastle
24|Fulham|Man City
24|Leeds|Bournemouth
24|Man United|Chelsea
24|Nott'm Forest|Brentford
25|Everton|Leeds
25|Coventry|Liverpool
25|Hull|Sunderland
25|Aston Villa|Bournemouth
25|Crystal Palace|Brentford
25|Newcastle|Chelsea
25|Fulham|Nott'm Forest
25|Tottenham|Man City
25|Man United|Brighton
25|Ipswich|Arsenal
26|Man City|Newcastle
26|Bournemouth|Crystal Palace
26|Brentford|Coventry
26|Chelsea|Ipswich
26|Liverpool|Hull
26|Arsenal|Fulham
26|Nott'm Forest|Man United
26|Brighton|Tottenham
26|Sunderland|Everton
26|Leeds|Aston Villa
27|Crystal Palace|Sunderland
27|Man United|Arsenal
27|Ipswich|Brighton
27|Coventry|Bournemouth
27|Everton|Nott'm Forest
27|Fulham|Leeds
27|Aston Villa|Chelsea
27|Tottenham|Liverpool
27|Newcastle|Brentford
27|Hull|Man City
28|Nott'm Forest|Newcastle
28|Bournemouth|Tottenham
28|Man City|Everton
28|Arsenal|Crystal Palace
28|Leeds|Hull
28|Sunderland|Man United
28|Liverpool|Aston Villa
28|Brighton|Fulham
28|Brentford|Ipswich
28|Chelsea|Coventry
29|Leeds|Brighton
29|Tottenham|Nott'm Forest
29|Crystal Palace|Fulham
29|Liverpool|Ipswich
29|Coventry|Man City
29|Bournemouth|Newcastle
29|Chelsea|Arsenal
29|Aston Villa|Hull
29|Man United|Everton
29|Sunderland|Brentford
30|Brentford|Bournemouth
30|Brighton|Coventry
30|Man City|Man United
30|Ipswich|Crystal Palace
30|Arsenal|Sunderland
30|Fulham|Liverpool
30|Hull|Chelsea
30|Nott'm Forest|Aston Villa
30|Everton|Tottenham
30|Newcastle|Leeds
31|Sunderland|Ipswich
31|Leeds|Nott'm Forest
31|Man United|Hull
31|Chelsea|Fulham
31|Liverpool|Newcastle
31|Coventry|Arsenal
31|Bournemouth|Man City
31|Crystal Palace|Everton
31|Tottenham|Brentford
31|Aston Villa|Brighton
32|Brighton|Chelsea
32|Fulham|Sunderland
32|Brentford|Leeds
32|Nott'm Forest|Liverpool
32|Hull|Coventry
32|Arsenal|Aston Villa
32|Ipswich|Man United
32|Man City|Crystal Palace
32|Newcastle|Tottenham
32|Everton|Bournemouth
33|Chelsea|Man City
33|Bournemouth|Arsenal
33|Tottenham|Hull
33|Man United|Crystal Palace
33|Everton|Brighton
33|Nott'm Forest|Sunderland
33|Newcastle|Ipswich
33|Aston Villa|Coventry
33|Brentford|Fulham
33|Leeds|Liverpool
34|Sunderland|Newcastle
34|Brighton|Nott'm Forest
34|Coventry|Man United
34|Hull|Bournemouth
34|Man City|Brentford
34|Fulham|Everton
34|Liverpool|Chelsea
34|Arsenal|Tottenham
34|Crystal Palace|Aston Villa
34|Ipswich|Leeds
35|Man City|Liverpool
35|Leeds|Arsenal
35|Fulham|Ipswich
35|Nott'm Forest|Crystal Palace
35|Brentford|Aston Villa
35|Newcastle|Coventry
35|Brighton|Sunderland
35|Tottenham|Chelsea
35|Everton|Hull
35|Bournemouth|Man United
36|Hull|Fulham
36|Man United|Leeds
36|Crystal Palace|Brighton
36|Aston Villa|Newcastle
36|Chelsea|Everton
36|Coventry|Tottenham
36|Sunderland|Bournemouth
36|Liverpool|Brentford
36|Ipswich|Man City
36|Arsenal|Nott'm Forest
37|Brentford|Hull
37|Newcastle|Crystal Palace
37|Tottenham|Man United
37|Leeds|Sunderland
37|Brighton|Liverpool
37|Nott'm Forest|Ipswich
37|Bournemouth|Chelsea
37|Fulham|Coventry
37|Everton|Arsenal
37|Man City|Aston Villa
38|Ipswich|Everton
38|Arsenal|Brighton
38|Liverpool|Bournemouth
38|Coventry|Nott'm Forest
38|Aston Villa|Tottenham
38|Hull|Newcastle
38|Man United|Fulham
38|Sunderland|Man City
38|Chelsea|Brentford
38|Crystal Palace|Leeds'''
fixtures=[]
for i,line in enumerate(fixture_text.strip().splitlines(),1):
    mw,h,a=line.split('|')
    fixtures.append({'MatchNumber':i+50,'Matchweek':int(mw),'HomeTeam':h,'AwayTeam':a})
fixtures=pd.DataFrame(fixtures)
assert len(fixtures)==330 and fixtures.groupby('Matchweek').size().eq(10).all()
# matchweek anchor dates (official schedule source; individual fixtures can be rescheduled)
anchors={6:'2026-10-10',7:'2026-10-17',8:'2026-10-23',9:'2026-10-31',10:'2026-11-07',11:'2026-11-21',12:'2026-11-28',13:'2026-12-02',14:'2026-12-05',15:'2026-12-12',16:'2026-12-19',17:'2026-12-26',18:'2026-12-29',19:'2027-01-01',20:'2027-01-05',21:'2027-01-16',22:'2027-01-23',23:'2027-01-30',24:'2027-02-06',25:'2027-02-10',26:'2027-02-20',27:'2027-02-27',28:'2027-03-03',29:'2027-03-13',30:'2027-03-20',31:'2027-04-10',32:'2027-04-17',33:'2027-04-24',34:'2027-05-01',35:'2027-05-08',36:'2027-05-15',37:'2027-05-23',38:'2027-05-30'}
fixtures['MatchweekStartDate']=pd.to_datetime(fixtures['Matchweek'].map(anchors))

# Current state: feed current results after historical 2025/26.
# Use actual chronological dates per MW anchor for state features, adding small within-week ordering via row order.
# To keep the state exact, create distinct dates based on matchweek anchor plus row-order hours.
cur_for_state=current_results.copy()
cur_for_state['MatchDate']=pd.to_datetime(cur_for_state['MatchDate']) + pd.to_timedelta(cur_for_state.groupby('Matchweek').cumcount(), unit='h')
combined_for_state=pd.concat([complete_hist[['Season','MatchDate','HomeTeam','AwayTeam','FullTimeHomeGoals','FullTimeAwayGoals','FullTimeResult']].copy(),cur_for_state[['Season','MatchDate','HomeTeam','AwayTeam','FullTimeHomeGoals','FullTimeAwayGoals','FullTimeResult']]], ignore_index=True)
combined_for_state=combined_for_state.sort_values('MatchDate').reset_index(drop=True)
_,future_state,future_h2h=build_state_features(combined_for_state)
# Note: final state also includes current 2026/27 results.

# Future rows features using final state (not current-match stats)
def future_row_feature(home,away,mwdate):
    hs=future_state.get(home,{'elo':1500,'hist':[],'home_hist':[],'last_date':None})
    as_=future_state.get(away,{'elo':1500,'hist':[],'away_hist':[],'last_date':None})
    hh,ah=hs['hist'],as_['hist']; hhh=hs['home_hist']; aah=as_.get('away_hist',[])
    # H2H from most recent 5 home-venue meetings
    ph=future_h2h.get((home,away),[])
    h2h_wr=mean_or_zero([x['home_win'] for x in ph]); h2h_dr=mean_or_zero([x['draw'] for x in ph]); h2h_gd=mean_or_zero([x['gd'] for x in ph])
    hrest=(pd.Timestamp(mwdate)-hs['last_date']).days if hs.get('last_date') is not None else 14
    arest=(pd.Timestamp(mwdate)-as_.get('last_date')).days if as_.get('last_date') is not None else 14
    hppg=ppg_last(hh); appg=ppg_last(ah); hwr=winrate_last(hh); awr=winrate_last(ah)
    hgf=mean_or_zero(hist_vals(hh,'gf')); agf=mean_or_zero(hist_vals(ah,'gf'))
    hga=mean_or_zero(hist_vals(hh,'ga')); aga=mean_or_zero(hist_vals(ah,'ga'))
    hgd=mean_or_zero([x['gf']-x['ga'] for x in hh]); agd=mean_or_zero([x['gf']-x['ga'] for x in ah])
    hv=ppg_last(hhh); av=ppg_last(aah)
    dt=pd.Timestamp(mwdate)
    return {
        'HomeTeam':home,'AwayTeam':away,'Home_Elo':hs['elo'],'Away_Elo':as_['elo'],'Elo_Difference':hs['elo']+50-as_['elo'],
        'Home_PPG_Last5':hppg,'Away_PPG_Last5':appg,'PPG_Difference':hppg-appg,
        'Home_WinRate_Last5':hwr,'Away_WinRate_Last5':awr,'WinRate_Difference':hwr-awr,
        'Home_GoalsFor_Last5':hgf,'Away_GoalsFor_Last5':agf,'GoalsFor_Difference':hgf-agf,
        'Home_GoalsAgainst_Last5':hga,'Away_GoalsAgainst_Last5':aga,'GoalsAgainst_Difference':hga-aga,
        'Home_GoalDiff_Last5':hgd,'Away_GoalDiff_Last5':agd,'GoalDiff_Difference':hgd-agd,
        'Home_HomePPG_Last5':hv,'Away_AwayPPG_Last5':av,'Venue_PPG_Difference':hv-av,
        'Home_RestDays':hrest,'Away_RestDays':arest,'Rest_Difference':hrest-arest,
        'H2H_HomeWinRate_Last5':h2h_wr,'H2H_DrawRate_Last5':h2h_dr,'H2H_GoalDiff_Last5':h2h_gd,
        'Month':dt.month,'DayOfWeek':dt.dayofweek
    }

future_X=pd.DataFrame([future_row_feature(r.HomeTeam,r.AwayTeam,r.MatchweekStartDate) for _,r in fixtures.iterrows()])
probs=final_xgb.predict_proba(final_prep.transform(future_X))
lr_future_gd=final_lr.predict(future_X)
fixtures_out=fixtures.copy()
fixtures_out['HomeWin_Probability']=probs[:,2]
fixtures_out['Draw_Probability']=probs[:,1]
fixtures_out['AwayWin_Probability']=probs[:,0]
fixtures_out['Prediction']=np.array(label_order)[np.argmax(probs,axis=1)]
fixtures_out['Confidence']=np.max(probs,axis=1)
fixtures_out['LR_Predicted_GoalDifference']=lr_future_gd
fixtures_out['LR_Implied_Result']=np.where(lr_future_gd>0.25,'H',np.where(lr_future_gd<-0.25,'A','D'))
fixtures_out.to_csv(BASE/'results/2026_27_all_upcoming_match_predictions.csv',index=False)

# Arsenal fixture spotlight
fixtures_out[(fixtures_out.HomeTeam=='Arsenal') | (fixtures_out.AwayTeam=='Arsenal')].to_csv(BASE/'results/arsenal_2026_27_remaining_predictions.csv',index=False)

# -------------------- Current table + Monte Carlo title race --------------------
teams=sorted(set(current_results.HomeTeam)|set(current_results.AwayTeam))
stand=defaultdict(lambda:{'P':0,'W':0,'D':0,'L':0,'GF':0,'GA':0})
for _,r in current_results.iterrows():
    h,a=r.HomeTeam,r.AwayTeam; hg,ag=r.FullTimeHomeGoals,r.FullTimeAwayGoals
    stand[h]['P']+=1; stand[a]['P']+=1; stand[h]['GF']+=hg; stand[h]['GA']+=ag; stand[a]['GF']+=ag; stand[a]['GA']+=hg
    if hg>ag: stand[h]['W']+=1; stand[a]['L']+=1
    elif hg<ag: stand[a]['W']+=1; stand[h]['L']+=1
    else: stand[h]['D']+=1; stand[a]['D']+=1
cur_rows=[]
for t in teams:
    d=stand[t]; cur_rows.append([t,d['P'],d['W'],d['D'],d['L'],d['GF'],d['GA'],d['GF']-d['GA'],3*d['W']+d['D']])
current_table=pd.DataFrame(cur_rows,columns=['Team','Played','W','D','L','GF','GA','GD','Points']).sort_values(['Points','GD','GF'],ascending=False).reset_index(drop=True)
current_table['Position']=np.arange(1,21)
current_table.to_csv(BASE/'results/2026_27_current_table_after_mw5.csv',index=False)

# Expected final table based on predicted probabilities
pts0=dict(zip(current_table.Team,current_table.Points))
exp_pts={t:float(pts0[t]) for t in teams}
for _,r in fixtures_out.iterrows():
    h,a=r.HomeTeam,r.AwayTeam
    exp_pts[h]+=3*r.HomeWin_Probability+1*r.Draw_Probability
    exp_pts[a]+=3*r.AwayWin_Probability+1*r.Draw_Probability
expected_tbl=pd.DataFrame({'Team':teams,'CurrentPoints':[pts0[t] for t in teams],'ExpectedFinalPoints':[exp_pts[t] for t in teams]})
expected_tbl=expected_tbl.sort_values('ExpectedFinalPoints',ascending=False).reset_index(drop=True)
expected_tbl['ExpectedRank']=np.arange(1,21)
expected_tbl.to_csv(BASE/'results/2026_27_expected_final_table.csv',index=False)

# Monte Carlo using XGBoost 1X2 probabilities, holding model state fixed at 2026-10-08.
rng=np.random.default_rng(42); sims=20000
team_idx={t:i for i,t in enumerate(teams)}
base_points=np.array([pts0[t] for t in teams],dtype=np.float32)
point_sims=np.tile(base_points,(sims,1))
# vectorized fixture-by-fixture update
for _,r in fixtures_out.iterrows():
    hidx=team_idx[r.HomeTeam]; aidx=team_idx[r.AwayTeam]
    u=rng.random(sims)
    ph,pd_prob,pa=r.HomeWin_Probability,r.Draw_Probability,r.AwayWin_Probability
    home_w=u<ph; draws=(u>=ph)&(u<ph+pd_prob); away_w=~(home_w|draws)
    point_sims[:,hidx]+=np.where(home_w,3,np.where(draws,1,0))
    point_sims[:,aidx]+=np.where(away_w,3,np.where(draws,1,0))
win_counts=np.zeros(len(teams),dtype=int)
finish_counts=np.zeros((len(teams),20),dtype=int)
for s in range(sims):
    # use expected goal-diff as a deterministic tie-break proxy for the points ordering
    order=np.argsort(-point_sims[s])
    win_counts[order[0]]+=1
    for pos,idx in enumerate(order): finish_counts[idx,pos]+=1
sim_tbl=pd.DataFrame({
    'Team':teams,
    'Title_Probability':[win_counts[team_idx[t]]/sims for t in teams],
    'Top4_Probability':[finish_counts[team_idx[t],:4].sum()/sims for t in teams],
    'Mean_Finish':[sum((i+1)*finish_counts[team_idx[t],i] for i in range(20))/sims for t in teams],
    'Expected_Final_Points':[point_sims[:,team_idx[t]].mean() for t in teams],
}).sort_values(['Title_Probability','Expected_Final_Points'],ascending=False).reset_index(drop=True)
sim_tbl['ProjectedRank']=np.arange(1,21)
sim_tbl.to_csv(BASE/'results/2026_27_title_race_monte_carlo.csv',index=False)

# Save predicted winner summary
winner=sim_tbl.iloc[0]
summary={'as_of':'2026-10-08','remaining_fixtures':len(fixtures_out),'simulations':sims,'model_winner':winner['Team'],'title_probability':float(winner['Title_Probability']), 'expected_final_points':float(winner['Expected_Final_Points'])}
json.dump(summary,open(BASE/'results/model_winner_summary.json','w'),indent=2)

# save combined historical state features for future reproducibility
future_X.to_csv(BASE/'results/2026_27_future_model_features.csv',index=False)

