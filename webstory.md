# Harmony on screen: Unraveling successful collaborations in the american movie industry

The movie industry, often hailed as the epitome of modern storytelling, stands as a captivating realm where creativity converges with technology, talent, and cultural expression. Spanning decades, this dynamic and ever-evolving landscape has become an integral part of global entertainment, captivating audiences and transcending borders. Beyond the enchanting narratives that unfold on the silver screen, the movie industry weaves a complex tapestry that extends its influence far beyond the confines of theaters.

At its core, the movie industry is a multifaceted ecosystem, comprising filmmakers, actors, writers, directors, producers, and a myriad of other creative minds, each contributing their unique talents to craft compelling stories. The collaborative nature of filmmaking fosters a delicate dance of ideas and visions, resulting in the birth of cinematic masterpieces that resonate with audiences on emotional, intellectual, and even spiritual levels. The relationships forged within the movie industry are as diverse as the stories it tells. Collaboration and synergy are at the heart of its success, with artists and technicians working in harmony to bring visions to life.

In this exploration of the movie industry, we will delve into the intricate relationships formed both on and off the screen, beyond traditional success metrics. From the creative collaborations that birth cinematic wonders to the societal impact that transcends the theater walls, join us on a journey through the enchanting world where imagination meets reality : the world of the movie industry.

--

In the ever-evolving realm of filmmaking, success is often attributed to the synergy between actors, scenarists, and directors. Our project seeks to unravel the intricate dynamics of this collaboration, going beyond traditional success metrics. By constructing comprehensive actor collaboration networks, we aim to explore the chemistry that arises from frequent co-starring and its impact on the success of movies. Our approach involves the creation of a web network that interconnects actors, scenarists, and directors, using the data from **CMU Movie Summary Corpus**.

## I don't like people, sorry
If you want to reduce the number of actors in your movie, unfortunately, that depends on your scenario. You can even have one actor playing all the roles ! However, in order to ensure its success, let's see which specific individual you should cast.

graph

AS we can see blablabla

--

## Tell me Johnny, who do you see often ?
These actors practically have each other on speed dial with their frequent collaborations.

## Let's play celebrity chess !
Get ready to play a round of celebrity chess as we unravel the patterns in Hollywood's grand game of actor collaboration networks. It's not just about moves on the board; it's a strategic dance of connections, rivalries, and who's stealing the spotlight on the chessboard. Let the games begin!

2
### Time traveling in the movie industry
4

### It's gossip time : who goes well together?
6
In our investigation to identify the key traits of successful actor clusters, we employed a series of t-tests to assess various attributes within our dataset, including gender disparity, the frequency of film collaborations between actor pairs, and several other factors. Notably, the only attribute that yielded a statistically significant result (with a p-value below 0.05) was the age difference between pairs of actors.


The accompanying plot organizes clusters based on their average ranks, with the more successful clusters (denoted by lower average ranks) positioned on the left, and the less successful ones (indicated by higher average ranks) on the right. It is important to emphasize that a lower average rank signifies greater success. The plot reveals a discernible trend: clusters with lower ranks tend to exhibit a marginally higher age disparity, suggesting a broader age range among members of these clusters. This variation in age could be indicative of the beneficial impact of seasoned actors, who potentially contribute greater experience and knowledge to the ensemble.

### Spice it up : The return of celebrity chess with even more collaboration
Adding directors and composers do not lead to a significant change in the collaborative network. As some well known directors are also actors, they are already part of the network. For composers, they are highly tied to one or a few directors. It is expected as directors tend to keep the same composer for their movies. When clustering the nodes only on directors and composers we find that it is almost always a director that make the bridge between clusters. There a lot of 2 nodes clusters composed of one director and one composer, which agrees with what we said before.

## It's a bit more twisted than your average plot twist : internal factors

We observe that for each figure, the average movie rating for the biggest category is around 6.45.

Both the age and film count difference has high variability when the difference is big due to the small number of movies. Otherwise it averages around 6.5 pretty consistently.

Whether it is the first film for one of the actors or for both does not have a significant impact on the average rating. The difference in average rating is around 0.1 in both cases.

When looking at the number of films a pair of actors have done together, we observe a small increase in the average rating up to 5 films. After that, the average rating decreases slightly and no particular trend emerges.

Two genres have a remarkable lower average rating than the rest: Softcore Porn and Space opera with 2.4 and 2.5 respectively. On the contrary, some genres neighbour the 8.0 average rating mark: Anti-war, Culture & Society, Film à clef, Tragedy and Reboot. The latter contains only three movies, thus it is less interesting. For the other, we have a least 50 movies per genre. Hence, these genres seem to be the most appreciated by the users.

7, 5
## What's life without even more complications : external factors
10
## Divining the future : it's crystal ball time
8

## Ending credits

# Research questions
1. **How does the frequency of actor co-starring impact the chemistry and success of collaborative films?**

2. **What patterns emerge when constructing and analyzing actor collaboration networks?**

3. **To what extent does the involvement of specific individuals (actors, scenarists, directors) contribute to the overall success of a collaborative project?**

4. **How does the collaborative network evolve over time, and are there recurring partnerships or clusters of individuals within the film industry?**

5. **What role does genre play in shaping collaborative dynamics, and how do successful collaborations differ across genres?**

6. **Are there specific collaborative subgraphs within the network that consistently yield successful movies, and what are the characteristics of these subgraphs?**

7. **How do factors such as creative differences, interpersonal relationships, and previous collaborative experiences impact the success of film projects within the identified networks?**

8. **Can the identified patterns and insights be applied to predict the potential success of future collaborative projects in the film industry?**

9. **In what ways does the inclusion of scenarists and directors in the collaborative network influence the overall success of a film, and are there notable differences in the impact of each role?**

10. **How do external factors, such as cultural trends or industry shifts, influence the dynamics of collaboration and the success of films within the identified networks?**

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