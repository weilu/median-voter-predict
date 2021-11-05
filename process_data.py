import pandas as pd
import numpy as np
import difflib
import re
from plotnine import *


state_raw = pd.read_csv('data/social_progressive_index.csv')
state_scores = []
for i, row in state_raw.iterrows():
    value = row.state_and_score
    matched = re.match(r'(.*)\(SPI score: (\d+\.\d+)\)', value)
    state_scores.append({
        'name': matched.group(1).strip(),
        'score': float(matched.group(2))
    })
state_scores_df = pd.DataFrame(state_scores)

state_meta_raw = pd.read_csv('data/us_states.csv')
state_meta_raw['name'] = state_meta_raw.name.map(lambda x: difflib.get_close_matches(x, state_scores_df.name)[0])
state_df = pd.merge(state_meta_raw, state_scores_df, on='name')
state_df.rename(columns={'score': 'social_progress_index'}, inplace=True)

house_raw = pd.read_csv('data/house_progressive_score.csv')
house_raw['state'] = house_raw.district.map(lambda d: d.split('-')[0])
house_df = house_raw.drop(columns='district')
house_df['senate'] = False

senate_raw = pd.read_csv('data/senate_progressive_score.csv')
senate_raw['senate'] = True
congress_df = house_df.append(senate_raw)

df = pd.merge(congress_df, state_df, left_on='state', right_on='code', how='left')
df.rename(columns={'crucial_vote_score': 'congressperson_progressiveness_score'}, inplace=True)
print(df)

def poly(x, degree=1):
    """
    Fit Polynomial

    These are non orthogonal factors, but it may not matter if
    we only need this for smoothing and not extrapolated
    predictions.
    """
    d = {}
    for i in range(degree+1):
        if i == 1:
            d['x'] = x
        else:
            d[f'x**{i}'] = np.power(x, i)
    return pd.DataFrame(d)

plot = (ggplot(df, aes(x='congressperson_progressiveness_score', y='social_progress_index', color='party'))
    + geom_point()
    + stat_smooth(
        method='lm',
        formula='y ~ poly(x, degree=2)',
    )
    + scale_color_manual(values={'R': 'red', 'D': 'darkblue', 'I': 'green'})
)
plot.save('social_progress_by_congress_progressiveness_parties.png', height=8, width=8)

plot = (ggplot(df, aes(x='congressperson_progressiveness_score', y='social_progress_index'))
    + geom_point()
    + stat_smooth(
        method='lm',
        formula='y ~ poly(x, degree=1)',
    )
)
plot.save('social_progress_by_congress_progressiveness.png', height=8, width=8)

