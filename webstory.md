# Harmony on screen: Unraveling successful collaborations in the american movie industry

The movie industry, often considered the pinnacle of modern storytelling, is a captivating realm where creativity meets technology, talent, and cultural expression. Across decades, this ever-evolving landscape has become a crucial element in global entertainment, captivating audiences worldwide and transcending geographical boundaries. Beyond the captivating narratives on the silver screen, the movie industry weaves a complex tapestry that extends its influence far beyond the confines of theaters.

At its core, the movie industry is a multifaceted ecosystem, comprising filmmakers, actors, writers, directors, producers, and numerous other creative minds. Each individual contributes unique talents to craft compelling stories. The collaborative nature of filmmaking fosters a delicate dance of ideas and visions, resulting in the creation of cinematic masterpieces that resonate with audiences on emotional, intellectual, and even spiritual levels. The relationships formed within the movie industry are as diverse as the stories it tells. Collaboration and synergy are at the heart of its success, with artists and technicians working in harmony to bring visions to life.

In this exploration of the movie industry, where success is the elusive golden ticket, we're diving headfirst into the dynamics between actors, scenarists, and directors. Brace yourself for a web network that connects the dots between actors, scenarists, and directors, courtesy of the **CMU Movie Summary Corpus** data. Get ready to unravel the intrigue, and discover how these Hollywood connections play a role in the make-it-or-break-it game of movie success. It's time to turn the spotlight behind the scenes, where imagination meets reality, and the movie industry takes center stage. Let the show begin! 🎬💫

## I don't like people, sorry
If you want to reduce the number of actors in your movie, unfortunately, that depends on your scenario. You can even have one actor playing all the roles ! However, in order to ensure its success, let's see which specific individual you should cast.

graph

AS we can see blablabla

--

## Tell me Johnny, who do you see often ?
These actors practically have each other on speed dial with their frequent collaborations. However, as we can observe in the plot below, the average Joe couldn't care less about their on-screen rendezvous. The frequency of actor co-starring doesn't impact their movie ratings. (+ de blabla technique?)

## Let's play celebrity chess !

2
### Time traveling in the movie industry
4

### 
Our quest to unveil the secrets of successful actor clusters involved a battery of t-tests. We dissected everything from gender dynamics to how often these stars collabprated on set. Notably, the only attribute that yielded a statistically significant result (with a p-value below 0.05) was the age difference between pairs of actors.


The accompanying plot organizes clusters based on their average ranks, with the more successful clusters (denoted by lower average ranks) positioned on the left, and the less successful ones (indicated by higher average ranks) on the right. Remember, lower ranks mean higher success ! And it appears that the groups with lower average ranks have a broader age range among them. In simpler terms, those seasoned actors bring more value to the group, possibly leveraging their wealth of experience and knowledge. Age before beauty, anyone?

Now that we've investigated the collaboration between actors, let's extend the scope of our story and add the collaborative networks of composers and scenarists.
 
### Spice it up : The return of celebrity chess with even more collaboration
Adding directors and composers do not lead to a significant change in the collaborative network. Some well-known directors pull double duty as actors, so they're already part of the network.

As for composers, they're practically tied at the hip with one or a select few directors. In fact, in the plot there are a lot of two-node squads, starring a director and a composer, as directors tend to keep the same composer for their movies. When clustering the nodes only on directors and composers, it is almost always a director that makes the bridge between clusters

## It's a bit more twisted than your average plot twist : internal factors

We observe that for each figure, the average movie rating for the biggest category is around 6.45.

Both the age and film count difference has high variability when the difference is big due to the small number of movies. Otherwise it averages around 6.5 pretty consistently.

Whether it is the first film for one of the actors or for both does not have a significant impact on the average rating. The difference in average rating is around 0.1 in both cases.

When looking at the number of films a pair of actors have done together, we observe a small increase in the average rating up to 5 films. After that, the average rating decreases slightly and no particular trend emerges.

Two genres have a remarkable lower average rating than the rest: Softcore Porn and Space opera with 2.4 and 2.5 respectively. On the contrary, some genres neighbour the 8.0 average rating mark: Anti-war, Culture & Society, Film à clef, Tragedy and Reboot. The latter contains only three movies, thus it is less interesting. For the other, we have a least 50 movies per genre. Hence, these genres seem to be the most appreciated by the users.

7, 5

## Divining the future : let's consult the crystal ball
8

Our initial exploration to predict the movie rating involved considering the following features defined in the context of pairs of actors that have played together in a movie:
- Age difference
- The difference in the number of movies they have played in
- The number of movies they have played together in
- Whether it was the first film for both actors or one of them
- The main genre of movies played by the actors

To unravel the cinematic enigma, we encoded the genre as a one-hot vector and scaled the numerical features. We then trained a linear regression model on the training set, corresponding of 80% of the data, and evaluated it on the test set. Alas, the model merely mustered an R2 score of 0.03 – a rather lackluster performance. Seeking further clarity, we subjected it to a 5-fold cross-validation, resulting in a mean R2 score of 0.01. The verdict? Predicting movie ratings with these specified features seems akin to reading an inscrutable crystal ball – not happening!

Undeterred, we plunged into a second act, computing a rank ratio born from normalized movie ratings and revenues, and accounting for inflation's dramatic influence. Here, a rank ratio edging towards 1 signaled a successful actor duo. Success, in our script, equated to a rank ratio of 0.6 or higher. The cast of features mirrored the first act, with a notable exception – the exclusion of a boolean feature signaling an actor's cinematic debut due to its high correlation with its boolean counterpart. (lequel???)

We trained a logistic regression model on the training set and evaluated it on the test set. The model achieved an accuracy of 0.78, coupled with a robust ROC-AUC of 0.82. To put our model in perspective, we concocted a baseline model that predictably leans toward the majority every time. It musters an accuracy of 0.58, and we can thus affirm that our model works well. Our crystal ball clears and the stage is set for success! 🌟🎥

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
