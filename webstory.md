# Harmony on screen: Unraveling successful collaborations in the american movie industry

The movie industry, often considered the pinnacle of modern storytelling, is a captivating realm where creativity meets technology, talent, and cultural expression. Across decades, this ever-evolving landscape has become a crucial element in global entertainment, captivating audiences worldwide and transcending geographical boundaries. Beyond the captivating narratives on the silver screen, the movie industry weaves a complex tapestry that extends its influence far beyond the confines of theaters.

At its core, the movie industry is a multifaceted ecosystem, comprising filmmakers, actors, writers, directors, producers, and numerous other creative minds. Each individual contributes unique talents to craft compelling stories. The collaborative nature of filmmaking fosters a delicate dance of ideas and visions, resulting in the creation of cinematic masterpieces that resonate with audiences on emotional, intellectual, and even spiritual levels. The relationships formed within the movie industry are as diverse as the stories it tells. Collaboration and synergy are at the heart of its success, with artists and technicians working in harmony to bring visions to life.

In this exploration of the movie industry, where success is the elusive golden ticket, we're diving headfirst into the dynamics between actors, scenarists, and directors. Brace yourself for a web network that connects the dots between actors, scenarists, and directors, courtesy of the **CMU Movie Summary Corpus** data. Get ready to unravel the intrigue, and discover how these Hollywood connections play a role in the make-it-or-break-it game of movie success. It's time to turn the spotlight behind the scenes, where imagination meets reality, and the movie industry takes center stage. Let the show begin! 🎬💫

## I don't like people, sorry
If you want to reduce the number of actors in your movie, unfortunately, that depends on your scenario. You can even have one actor playing all the roles ! However, in order to ensure its success, let's see which specific individual you should cast.

graph

AS we can see blablabla

## Tell me Johnny, who do you see often ?
These actors practically have each other on speed dial with their frequent collaborations. However, as we can observe in the plot below, the average Joe couldn't care less about their on-screen rendezvous.
<iframe src="graphs/average_rating_by_nb_films_together.html"></iframe>

To confirm this data, using a linear regrassion model, R2 is equal to a measly 0.001, and thus the frequency of actor co-starring doesn't impact their movie ratings.

## Let's play celebrity chess !
### Time traveling in the movie industry
4

We have decided that only collaborations occurring at least three times will be featured in our analysis. This criterion is based on the rationale that in the movie industry, a partnership isn't typically recognized as a significant "duo" if they have only collaborated once or twice. Our focus is on observing the development of these collaborations over time and how they form clusters, reflecting either successful or less favorable outcomes. Initially, our network will encompass films released from 1980 to 1995. Subsequently, we will extend the timeline up to 2010, and finally, we will consider a broad range from 1980 to 2023, resulting in a more intricate network.

The collaborations are assessed using two primary ranking criteria. The first method ranks duos based on the average rating of their joint projects. In cases where ratings are identical, the average revenue is used as a secondary factor to differentiate the ranks. Duos with the same values in both criteria will receive the same rank. The second method prioritizes ranking based on revenue, followed by average ratings. The duo's position in these rankings will be indicative of their performance level and will influence their representation in the network graph.

This is the legend of our graph : 

Nodes: These symbolize the actors. The size of a node correlates with the number of unique collaborations an actor has engaged in.

Edges: These signify collaborations between pairs of actors. The thickness of an edge reflects the frequency of collaborations between the actors involved.

Color Scheme: This aspect denotes the success level of collaborations. Collaborations are then organized based on two criteria: rating and film revenue. The ranking determines the color of the edge, with higher ranks resulting in pinker edges and lower ranks leading to browner edges. For an actor (node), their color is a composite of the shades from all the edges (collaborations) they have participated in.


For the First Network Based on Movie Ratings:

In this network, it's evident that clusters of actors form rapidly. Actors who begin working together often continue to collaborate, leading to increasingly prominent connections within these groups. This phenomenon is clearly visible with the expansion of edges within clusters. Notably, smaller clusters, comprising fewer than four actors, generally exhibit lower performance. This could be attributed to successful actors gravitating towards larger, more established groups. Alternatively, a longitudinal view reveals a pattern where prominent actors within major clusters frequently collaborate with newcomers. This inclusion typically results in the newcomer gaining significant prominence within the cluster. These clusters often display similar colors, indicating a consistent level of success, likely because they originate from actors frequently cast in the same series or films. 

For the Network Based on Revenue Ratings:

The cluster formations in this network are largely similar to the first, but with notable differences in the color scheme employed. Larger clusters tend to exhibit greener hues, suggesting higher profitability. Contrarily, large clusters with brown coloring are rare and mostly found among smaller groups with fewer actors. This pattern suggests a correlation between the length and success of an actor's career and their network's size: a broader network implies more varied collaborations, potentially leading to roles in higher-grossing films.

### It's gossip time...
Our quest to unveil the secrets of successful actor clusters involved a battery of t-tests. We dissected everything from gender dynamics to how often these stars collabprated on set. Notably, the only attribute that yielded a statistically significant result (with a p-value below 0.05) was the age difference between pairs of actors.

<iframe src="graphs/"></iframe>
The accompanying plot organizes clusters based on their average ranks, with the more successful clusters (denoted by lower average ranks) positioned on the left, and the less successful ones (indicated by higher average ranks) on the right. Remember, lower ranks mean higher success ! And it appears that the groups with lower average ranks have a broader age range among them. In simpler terms, those seasoned actors bring more value to the group, possibly leveraging their wealth of experience and knowledge. Age before beauty, anyone?
6

### Spice it up : The return of celebrity chess with even more collaboration
Now that we've investigated the collaboration between actors, let's extend the scope of our story and add the collaborative networks of composers and scenarists. As you can see in the plot below, adding directors and composers do not lead to a significant change in the collaborative network. Some well-known directors pull double duty as actors, so they're already part of the network.

As for composers, they're practically tied at the hip with one or a select few directors. In fact, in the plot there are a lot of two-node squads, starring a director and a composer, as directors tend to keep the same composer for their movies. When clustering the nodes only on directors and composers, it is almost always a director that makes the bridge between clusters.


## It's a bit more twisted than your average plot twist
### Genre investigation
In our first analysis, we meticulously calculated the ratios of movie genres for each cluster, basing our calculations on the films in which actors within the cluster participated. This approach provided a collective genre profile for each cluster. Conversely, our second analysis adopted a different perspective, focusing on the main genres associated with the actors themselves. This method offered an individualistic genre representation of each cluster. Upon initial inspection of both datasets, there wasn't an immediately discernible correlation between specific genres and the clusters' ranks.

The analysis took a compelling turn when we applied T-tests to each genre across the clusters. A significant finding emerged from this statistical examination, particularly with the crime-fiction genre. When analyzed in the context of the actors' main genres, crime-fiction yielded a p-value below 0.05, indicating a statistically significant association. Delving deeper into this discovery, especially concentrating on the crime-fiction genre, revealed a notable pattern: an increased prevalence of this genre corresponded with lower cluster rankings, suggesting a potential inverse relationship.

Extending this statistical scrutiny to the movie genre ratios, we observed a parallel trend for crime-fiction, which again manifested a p-value under 0.05. This consistency in results underscores the genre's impact on cluster rankings. However, the study uncovered an intriguing contrast with genres such as "fantasy" and "fantasy adventure," where the p-values were less than 0.05. This intriguing outcome implies that unlike crime-fiction, which appears to negatively influence a cluster's average rank, the presence of fantasy or fantasy adventure genres could contribute positively to enhancing a cluster's rank.

The significant correlation between the prevalence of crime-fiction and lower cluster rankings suggests that clusters heavily skewed towards this genre might face certain challenges. This could be attributed to the demanding nature of crime-fiction narratives, which often require intense, dramatic performances and may not always appeal to a broad audience. Such a genre-specific focus could limit the versatility and appeal of the cluster's collective filmography.

On the other hand, the positive correlation observed with fantasy and fantasy adventure genres hints at a different dynamic. These genres, often characterized by imaginative storytelling and visual spectacle, might offer more opportunities for creative expression and broad audience appeal. This can lead to a more diverse and engaging portfolio of work, potentially contributing to higher cluster rankings.

These patterns underscore the idea that the collective genre profile of a cluster is not just a reflection of individual actor preferences, but a strategic element that can influence the cluster's overall success. The study suggests that a balanced and diverse genre portfolio within a cluster might be more conducive to achieving higher rankings, possibly due to wider audience appeal and greater opportunities for showcasing a range of acting skills and storylines. This insight could be valuable for actors and industry professionals in forming collaborative groups and choosing projects, emphasizing the importance of genre diversity as a strategic consideration in their career trajectories.

### ?
We observe that for each figure, the average movie rating for the biggest category is around 6.45.

Both the age and film count difference has high variability when the difference is big due to the small number of movies. Otherwise it averages around 6.5 pretty consistently.

Whether it is the first film for one of the actors or for both does not have a significant impact on the average rating. The difference in average rating is around 0.1 in both cases.

When looking at the number of films a pair of actors have done together, we observe a small increase in the average rating up to 5 films. After that, the average rating decreases slightly and no particular trend emerges.

Two genres have a remarkable lower average rating than the rest: Softcore Porn and Space opera with 2.4 and 2.5 respectively. On the contrary, some genres neighbour the 8.0 average rating mark: Anti-war, Culture & Society, Film à clef, Tragedy and Reboot. The latter contains only three movies, thus it is less interesting. For the other, we have a least 50 movies per genre. Hence, these genres seem to be the most appreciated by the users.

7, 5

## Divining the future : let's consult the crystal ball
8

Our exploration to predict the movie rating considered the following features defined in the context of pairs of actors that have played together in a movie:
- Age difference
- The difference in the number of movies they have played in
- The number of movies they have played together in
- The main genre of movies played by the actors

To unravel the cinematic enigma, we computed a rank percentile born from normalized movie ratings and revenues and accounting for inflation's dramatic influence. Here, a rank percentile edging towards 1 signals a successful actor duo. We have considered a successful movie to be successful if its ranking was in the 40% more successful movies, which, in our script, equates to a rank percentile of 0.6 or higher. Using the cast of features above, we trained a logistic regression model on the training set (80% of the data) and evaluated it on the test set. Our crystal ball is ready and the stage is set for success! 🌟🎥

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
