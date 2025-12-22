# MatchPick
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
3.  **Output:** The system assigns an "Excitement Score" or classifies the match into tiers (e.g., "Must Watch", "Solid Choice", "Likely Boring").


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

* **Weighted Analysis (Regression):** To calculate a specific "Excitement Score", we consider methods like linear regression for interpreting clear variable weights, and neural networks to capture complex, non-linear patterns in match momentum.
* **Classification:** To categorize matches into specific tiers (Tier 1: Thriller, Tier 2: Watchable, Tier 3: Slow), we explore algorithms such as k-nearest neighbors to find similar historical matches.


## Demo/Example/Experiment
As a proof of concept for this course, I have included a Python script (`main.py`) that implements a simplified version of this logic.
* It uses **Linear Regression / Basic Scoring** based on historical goal averages from `Football-Data.co.uk`.
* It demonstrates the core principle: taking raw data and converting it into a ranked recommendation list for the user.
  
```
code

```
  

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
