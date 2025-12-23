# XciteFootball
Final project for the Building AI course


## Summary
This project uses AI to predict which football matches of the current gameweek will be the most entertaining and fun to watch. By analyzing historical stats like goal averages, big chances and play intensity, it ranks upcoming fixtures to help fans decide which games are worth watching.


## Background
Football fans often face a dilemma on weekends: there are too many matches and limited time to watch them. Often, we choose to watch a "big game" based on team reputation, only to be disappointed by a boring, defensive match, while missing a thrilling high-scoring game elsewhere.
* **Problem:** Time is wasted watching boring matches based on biased intuition or marketing. This can lead to viewer fatigue and a gradual decline in audience engagement over the long term.
* **Frequency:** Occurs every matchday (weekly) for millions of fans.
* **Personal Motivation:** As a football fan, I want to optimize my viewing time by prioritizing matches with the highest probability of action, goals, and intensity.
  
This project solves this by replacing intuition with data-driven probability.


## How is it used?
The solution is designed for football fans who want to curate their weekend viewing schedule.
1.  **Input:** The user (or the system) inputs the upcoming match fixtures for the week.
2.  **Processing:** The AI analyzes the recent form of both teams (goals scored/conceded, big chances created/conceded, defensive rigidity, style of play, etc.) from historical data, among many other extra factors or parameters.
3.  **Output:** The system assigns an "Excitement Score" or classifies the match into tiers (e.g., "Must Watch", "Watchable", "Likely Boring").


## Data sources and AI methods
To accurately predict the entertainment value of a match, the project is designed to process a diverse range of data points beyond simple scores.

**Data Sources & Parameters:**
The model aims to integrate multiple layers of data:

* **Performance Metrics:** Expected Goals (xG), Big Chances created, and defensive errors.
* **Match Dynamics:** Real-time momentum charts (like Sofascore's "Attack Momentum") to measure intensity.
* **Contextual Factors:** Weather conditions (rain/snow), stadium atmosphere (attendance/rivalries), and team motivation (relegation battle, title race...).
* **Player Quality:** Individual player ratings and availability (injuries to key stars).

**AI Techniques:**
The comprehensive solution envisions a multi-modal machine learning model that weighs these diverse inputs using a combination of techniques:

* **Weighted Analysis (Regression):** To calculate a specific "Excitement Score", we consider methods like linear or logistic regression for interpreting clear variable weights, and neural networks to capture complex, non-linear patterns in match momentum.
* **Classification:** To categorize matches into specific tiers (Must Watch, Watchable, Likely Boring), we explore algorithms such as k-nearest neighbors to find similar historical matches.


## Demo/Example/Experiment
As a proof of concept for this course, I have included a Python script that implements a simplified version of this logic. This simplified example assumes "Entertainment=Quantity of goals". This script analyzes the current season's performance by calculating the average goals scored and conceded for each team based on their available historical match data. It trains a Logistic Regression model to predict the probability of a game ending with over 2.5 goals, using this metric to classify the match as "Exciting." Finally, the system processes the upcoming fixtures and outputs a ranked list, assigning a specific entertainment score from 0 to 10 to help fans prioritize which games to watch. Note: To ensure accuracy, the first few games of the season are excluded from the training set, as early averages lack the necessary volume to be statistically significant. 

To run this experiment, you need Python and the following libraries:
`pip install pandas scikit-learn`

<details>
<summary><strong>Click here to view the Python Code</strong></summary>

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# --- 1. DATA LOADING & PREPARATION ---
DATA = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"  # Find the current or any LaLiga season stats CSV in https://www.football-data.co.uk/spainm
try:
    df = pd.read_csv(DATA)
    # Converts 'Date' column to datetime objects (handling day/month/year format)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    print(f"Data loaded successfully: {len(df)} matches found.")
except FileNotFoundError:
    print(
        f"Error: Could not find {DATA}. Please download it from https://www.football-data.co.uk/englandm.php"
    )
    exit()


# --- 2. HELPER FUNCTION: HISTORICAL STATS ---
MIN_GAMES_REQUIRED = 5  # This is used to skip early season matches to ensure goal averages are statistically significant.


# Calculates the average goals scored/conceded by a team just before a specific match date.
def get_team_stats_at_date(team, current_match_date, full_df):

    # Gets all games played by this team before the current match date
    past_games = full_df[
        ((full_df["HomeTeam"] == team) | (full_df["AwayTeam"] == team))
        & (full_df["Date"] < current_match_date)
    ]

    # Checks if the team has enough history to be analyzed and skips the match for training
    if len(past_games) < MIN_GAMES_REQUIRED:
        return None

    goals_scored = 0
    goals_conceded = 0
    count = 0

    # Calculates totals
    for _, game in past_games.iterrows():
        if game["HomeTeam"] == team:
            goals_scored += game["FTHG"]  # Full Time Home Goals
            goals_conceded += game["FTAG"]  # Full Time Away Goals
        else:  # Played Away
            goals_scored += game["FTAG"]
            goals_conceded += game["FTHG"]
        count += 1

    # Calculates Averages
    avg_scored = goals_scored / count
    avg_conceded = goals_conceded / count

    return [avg_scored, avg_conceded]


# --- 3. BUILDING THE TRAINING DATASET ---

X_train = []  # Features (Stats)
y_train = []  # Target (Labels)
skipped_matches = 0

for index, row in df.iterrows():
    # Gets stats for both teams exactly as they were before the match started
    home_stats = get_team_stats_at_date(row["HomeTeam"], row["Date"], df)
    away_stats = get_team_stats_at_date(row["AwayTeam"], row["Date"], df)

    # If any team doesn't have enough history (first weeks of the season), skips
    if home_stats is None or away_stats is None:
        skipped_matches += 1
        continue

    # Combines stats: [Home_Avg_Scored, Home_Avg_Conceded, Away_Avg_Scored, Away_Avg_Conceded]
    features = home_stats + away_stats
    X_train.append(features)

    # Defines Target: 1 if match was "Exciting" (Over 2.5 goals), 0 if "Boring"
    total_goals = row["FTHG"] + row["FTAG"]
    label = 1 if total_goals > 2.5 else 0
    y_train.append(label)

# --- 4. MODEL TRAINING ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- 5. PREDICTION FUNCTION FOR FUTURE GAMES ---
current_date = datetime.now()


def get_match_prediction(local, visitor):
    # Gets the most up-to-date stats for today
    h_stats = get_team_stats_at_date(local, current_date, df)
    a_stats = get_team_stats_at_date(visitor, current_date, df)

    if h_stats is None or a_stats is None:
        print(
            f"Insufficient data for {local} or {visitor} (Played less than {MIN_GAMES_REQUIRED} games)."
        )
        return None

    # Prepares input for the model
    input_data = [h_stats + a_stats]

    # Gets probability of class '1' (Exciting Match)
    prob = model.predict_proba(input_data)[0][1]
    score = prob * 10

    return {"local": local, "visitor": visitor, "score": score}


# --- 6. PREDICTING NEXT MATCHWEEK ---
print("\n" + "=" * 80)
print(
    f"AI PREDICTIONS: UPCOMING MATCHWEEK (Sorted by Score)(Based on {len(X_train)} historical matches)"
)
print("=" * 80)

# --- UPDATE FIXTURES HERE ---
# Replace the list below with the actual upcoming matches you want to predict. IMPORTANT: Ensure team names match EXACTLY with the ones in the CSV.

fixtures = [
    ("Espanol", "Barcelona"),
    ("Vallecano", "Getafe"),
    ("Celta", "Valencia"),
    ("Osasuna", "Ath Bilbao"),
    ("Elche", "Villarreal"),
    ("Sevilla", "Levante"),
    ("Real Madrid", "Betis"),
    ("Mallorca", "Girona"),
    ("Alaves", "Oviedo"),
    ("Sociedad", "Ath Madrid"),
]

# List to store valid predictions
predictions_list = []

for h, a in fixtures:
    try:
        result = get_match_prediction(h, a)
        if result:
            predictions_list.append(result)
    except Exception as e:
        print(f"Error processing {h} vs {a}: {e}")

# --- SORTING & PRINTING ---
# Sorts the list by 'score' in descending order
predictions_list.sort(key=lambda x: x["score"], reverse=True)

# Prints the sorted results
for p in predictions_list:
    print(f"{p['local']:<20} vs {p['visitor']:<20} | Score: {p['score']:.1f}/10")

```
</details>

## Challenges
While the model improves decision-making, it cannot predict the unpredictable nature of sports.

* **Unpredictable Events:** A red card in the 5th minute or a sudden injury can completely change the dynamic of a game, which pre-match statistics cannot foresee.
* **Subjectivity:** "Fun" is subjective. Some fans enjoy a tactical defensive battle (0-0), while this model prioritizes goals and offensive stats.
* **Ethical Considerations:** If this tool were used for gambling or betting, it would require strict disclaimers. This project is purely for entertainment purposes and optimizing leisure time, not for financial advice.


## What next?
To transform this vision into a fully functional product, the next steps focus on implementation and automation:

* **From Static to Real-Time:** The biggest leap involves moving from analyzing historical CSV files to building live data pipelines. This means automating the collection of complex variables (weather, real-time momentum, lineups) via APIs so the "Excitement Score" is generated instantly before kickoff.
* **User Feedback Loop:** The project could grow by allowing users to vote on whether they enjoyed a recommended match. This data would feed back into the model to refine its accuracy.
* **Personalization:** Allowing users to manually customize the algorithm by adjusting the weights of parameters (e.g., valuing "high scoring" more than "tactical intensity") and selecting favorite teams or players to boost their specific matches in the final ranking.
* **Skills & Assistance Needed:** Domain expertise in football analytics is crucial to correctly perform feature selection and assign the right weight to the variables that truly matter. Additionally, implementing advanced Web Scraping and Cloud Deployment pipelines is essential.

## Acknowledgments
* Historical match data provided freely by [Football-Data.co.uk](https://www.football-data.co.uk/).
* This project was created as a final exercise for the [Building AI](https://buildingai.elementsofai.com/) course.
