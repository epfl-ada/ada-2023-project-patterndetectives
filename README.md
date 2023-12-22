# Harmony on screen: Unraveling successful collaborations in the american movie industry

**Team members**: [Adrien Vauthey](https://github.com/Lugiasc), [Kaan Uçar](https://github.com/Kaan-wq), [Elias Naha](https://github.com/Eliasepfl), [Keren Bensoussan](https://github.com/sdjfkjsh), [Julien Ars](https://github.com/merlebleue)

**Data zip folder to drop in the project :** https://drive.google.com/file/d/1H5PKd_VSk6jRUzvEh8h3AlxLpLpTcbMr/view?usp=drive_link

**The main notebook is `main.ipynb`**

# Table of contents
- [Abstract](#abstract)
- [Research questions](#research-questions)
- [Proposed dataset](#proposed-dataset)
- [Methods](#methods)
- [Proposed timeline](#proposed-timeline)
- [Organization within the team](#organization-within-the-team)
- [Contributions of each team member](#contributions-of-each-team-member)

# Abstract
This project aims to delve into the intricate dynamics of successful collaboration in the film industry, extending beyond conventional success metrics to explore the chemistry between actors through frequent co-starring. The approach involves constructing actor collaboration networks, interconnecting actors, scenarists, and directors within a comprehensive web network. Employing a granular methodology, the web network is subdivided based on movie genres, allowing for the analysis of collaborative subgraphs. The focus is on identifying patterns and relationships that contribute to the success of movies within each genre. This research not only provides insights into individual performances but also offers an understanding of collaborative networks. By incorporating diverse talents and considering genre-specific nuances, the project aims to yield valuable insights for future filmmaking collaborations, potentially influencing industry practices and enhancing the likelihood of successful movie projects. The use of network analysis tools adds a visual dimension, aiding in the interpretation of the complex web of collaborations in the film industry.

# Research questions, replaced in the order of the data story 
## A Lone Figure in the Filmic Expanse: Assessing Personal Influence and Prolificacy

1. **To what extent does the involvement of specific individuals (actors, composers, directors) contribute to the overall success of a collaborative project?**

2. **How does the frequency of actor co-starring impact the chemistry and success of collaborative films?** 

## From Solo Endeavors to Collective Synergy: The Transformation through Alchemy and Networks

3. **What patterns emerge when constructing and analyzing actor collaboration networks, and how do these patterns differ across various movie genres?**\
=> This question has been merged with the question 6 because of similarity and is partially answered there

4. **How does the collaborative network evolve over time, and are there recurring partnerships or clusters of individuals within the film industry?**

5. **Are there specific collaborative subgraphs within the network that consistently yield successful movies, and what are the characteristics of these subgraphs?**

6. **In what ways does the inclusion of composers and directors in the collaborative network influence the overall success of a film, and are there notable differences in the impact of each role?**

## Deciphering the Dynamics of Film Collaboration: Impact, Genre Influence, and Forecasting Success

7. **How do factors such as creative differences, interpersonal relationships, and previous collaborative experiences impact the success of film projects within the identified networks?**

8. **What role does genre play in shaping collaborative dynamics, and how do successful collaborations differ across genres?**

9. **Can the identified patterns and insights be applied to predict the potential success of future collaborative projects in the film industry?**

10. **How do external factors, such as cultural trends or industry shifts, influence the dynamics of collaboration and the success of films within the identified networks?**\
=> The researches for this question ended really deep, and we didn't find any relevant results so this question will be left unanswered.

Addressing these research questions will likely provide a comprehensive understanding of the collaborative dynamics in the film industry and offer practical insights for future filmmaking endeavors.

# Proposed dataset
[IMDb dataset](https://developer.imdb.com/non-commercial-datasets/): Used several datasets from IMDb to create a single dataset that is used to populate our own datasets with more information.
* [IMDb title](https://datasets.imdbws.com/title.basics.tsv.gz): Contains the basic information of movies.
* [IMDb crew](https://datasets.imdbws.com/title.crew.tsv.gz): Contains the directors and scenarists of movies.
* [IMDb ratings](https://datasets.imdbws.com/title.ratings.tsv.gz): Contains the ratings of movies.
* [IMDb name](https://datasets.imdbws.com/name.basics.tsv.gz): Contains the basic information of actors, directors and scenarists.
* [IMDb principals](https://datasets.imdbws.com/title.principals.tsv.gz): Contains information of people involved in movies.

# Methods
### 1. Database manager
Creation of the database and the population of the database with the data from the IMDb dataset.

- Merging database: Merge the IMDb datasets into a single dataset that will be used to populate the database.
- Cleaning data: Clean the data from the IMDb dataset to remove duplicates and to remove data that is not relevant to the project.

### 2. Analysis
Analysis of the data to answer the research questions using the CMU movie corpus.

- Processing data: Process the data from the corpus.
    - Character data: Analysis of the character dataset
    - Movie data: Analysis of the movie dataset
    - Revenue analysis around month of release: Categorize each movie into separate bins based on two criterias. Firstly, the first two genres they appear to belong to, and secondly, the month in which they were released. 

- Merging all databases: Merge databases to start performing some analysis on success of a collaboration between actors.

### 3. Transformation around data.
- Inflation process: Adjust the revenue of each movie to the inflation of the year it was released in.
- Creation of new features on the merged database: Creation of new features to help understand the evolution of actors individually.
- Creation of a database of pairs of actors: Creation of a database of pairs of actors that have worked together in a movie.

### 4. Actor network
**Nodes**: they represent actors, the bigger a node is, the more the actor has made different collaborations. \
**Edges**: they represent a collaboration between a pair of actors, the size of the edge represents the number of collaborations they have made

### 5. Propensity score matching
Propensity score matching to predict the revenue of the movie using a linear regression.

# Proposed timeline
**Before milestone 2: Define metrics and parameters**
- Refine and finalize the success metrics for movies.
- Determine parameters for measuring chemistry and collaboration success.
- Identify specific criteria for categorizing successful collaborations.

**Week 1: Construct collaborative networks**
- Utilize the cleaned data to construct actor collaboration networks.
- Establish links between actors, scenarists, and directors based on their collaborative history.
- Apply network analysis techniques to identify central nodes and clusters.

**Task 1: Genre-specific subnetworks**
- Implement the granular approach by dividing the collaborative network into subnetworks based on movie genres.
- Analyze and visualize genre-specific subnetworks.
- Identify patterns and relationships within each genre.

**Task 2: Analyze collaborative subgraphs**
- Focus on collaborative subgraphs within each genre to determine their impact on successful movies.
- Explore recurring partnerships and their influence on success.
- Examine how external factors may affect specific subgraphs.

**Task 3: Prepare the datastory website**
- Explore the possibilities for a website using Jekyll
- Prepare the layout of the website
- Implement the analysis findings into the website

**Week 4 : Finalize analysis and the datastory**
- Synthesize findings from genre-specific analyses.
- Draw conclusions and insights.
- Review the prepared datastory and make changes if necessary.

Week 1 and 4 will be done by all team members. Task 1, 2 and 3 will be done by different group members (2 people for Task 1, 2 people for Task 2, and 1 for Task 3), and should be finished by the end of week 3.

# Organization within the team
| Task   | Responsible |
| :--- | :--- |
| Task 1 | Adrien and Julien |
| Task 2 | Elias and Kaan |
| Task 3 | Keren |
| Everythin else | Whole team |

The collaboration plan was respected, everyone answered their question except for the last question which was deemed too hard.

# Contributions of each team member
- Adrien Vauthey: Worked on some of the data analysis, created the dataframe of pairs of actors, answered questions 6, 7 and 9.

- Kaan Uçar: Augmented the original data, worked on some of the data analysis, answered questions 5 and 8.

- Elias Naha: Worked on some of the data analysis, implemented the inital code for the creation of the network, created the dataframe of movies, answered questions 4 and 5.

- Keren Bensoussan: Worked on the website, answered question 1 and 2 with Julien and wrote the datastory based on the answers of the other members of the group.

- Julien Ars: Worked on the website, answered question 1 and 2 with Keren and wrote the datastory based on the answers of the other members of the group.
