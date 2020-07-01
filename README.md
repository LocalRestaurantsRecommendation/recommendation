# Recommendation Function Write Up

The goal of this project is to help uncover user's interests in local restaurants he /she might not know yet. 

This project is target for foodies like me :) 

This is internal project for fullstack website service: local restaurants recommendations.


## Recommendation Function
I can analyze it in different angles.

### Goal / Input / Output
The output of my recommendation function is to recommend a user top 10 nearby restaurants users are most likely to like.

The input is known user-item score from user interactions.

Options I considered were:
* User rate on some restaurant in scale ```1 - 5```
* User clicked on some restaurant ```>= 1``` times
* User saved some restaurants on the favorite list / shared with friends

Due to the limited time, I only used users' rating provided by [Yelp Open Dataset](https://www.yelp.com/dataset).

The goal is to get higher precision and recall on output.
This is evaluated on data we already know, by user-based average precision @ 10.

### Candidate Retrival / Score / Re-rank
I will cover these steps in details later on, here we talked about high levels.

For candidates on input user, I only consider nearby restaurants, which the user never interact with.

For scoring, I tried several models:
1. baseline model
2. ALS model
3. Naive hybrid version of basline & ALS model
4. Time biased hybrid model

For more details on each model, please refer to evaluation part.

For Re-rank, top 10 restaurants are returned in descending order of scores.

## Data Source

Because I started from no users or data, I used the [Yelp Open Dataset](https://www.yelp.com/dataset) as the starting point of training my recommendation function.

The dataset includes users' rating behavior on local business, which includes restaurants. As I only care about the subset of the data, I did the litte preprocess and I got

| Data Name              | Data Num |
|:----------------------:|:--------:|
| User                   | 1256822  |
| Restaurants            | 36875    |
| Reviews On Restaurants | 3951722  |

### Data Understanding
Other interesting facts I learned from the data.

* there are ```683``` cities in total

### Data Preprocess
One question for recommending restaurants is that we can have restaurants open and close often. It's not reasonable to recommend a closed restaurant, and my recommendation function should be aware of new restaurants as soon as possible.

My solution is to use a finite set of feature vectors to cover all kinds of restaurants, so that once a new restaurant comes in, I know exactly what feature vector the restaurant has.

I used the feature vectors to represent the set of restaurants also in my recommendation function.

The feature vectors I choosed was developed from set of features a restaurant can have in Yelp Open Dataset. For each restaurant, I split useful information into 3 categories:

* True/False features: set of features a restaurant can be true or false 
* Value features: set of features maps to a fixed set of string values, appended with my special "none" value
* Exist features: set of features I only care if the feature label exist or not

| Feature Name     | Feature Num |
|:----------------:|:-----------:|
| T/F feature      | 50          |
| Val feature      | 46          |
| Exist feature    | 750         |

### More On Preprocess
So far I mapped between restaurants and feature vectors, here is some interesting fact I learned from this mapping:

* there are ```36341``` feature vectors used in original dataset
* there are ```353``` feature vectors used for multiple restaurants
* At most ```18``` restaurants have exactly same feature vector

### Resolve Disagreement for User and Feature Pair

Since multiple restaurants can have same feature vector, it's possible for a user to have conflict rating on save feature vector.

From my dataset,
* Max rating difference between single user-feature vector pair is ```4```
* In total, ```99``` user-feature vector pairs' rating difference ```> 2```
* The disagreement happens in ```57``` feature vectors

Consider in total of ```36341``` feature vectors, ```57``` is fairly small number I can tolerant. So here I didn't step back to re-design the feature vectors. Instead, I resolve the disagreement in a simple way:

For all conflict pairs, the rating behaviors are replaced by following new rating:

```
rating = average rating over conflict pairs
timestamp = latest timestamp over conflict pairs
```

That is, I simply consider those restaurants "same".

## How To Train Data

### What Data Should I Train?
It only makes sense that I got data now to predict future. So I did chronologic split on each user.

But how many data should I take from each user? 

My solution is to do personalized training, which is to have a model for each user.

Here we define several variables:

* ```min_rating``` - threshold to filter out users we predict on
* ```train_number``` - the number of rating behvaiors we know about a user in train set

I started with ```min_rating >= 300```, I gained precision and recall on those users.

For each user, I need to choose a train_number. 

For example, I choosed number ```6``` on user A, who rated ```20``` restaurants in total.

That is, I picked the first ```6``` ratings chronologically to train the personalized model to predict the user's preference on other restaurants.

In collaborative filtering method, I also need other users' behvaiors. Note that my training set contains all the information I can use to train the model.

I picked the timestamp when user A gave the ```6th``` rating, and used that to filter out ratings for other users that should be in my training set.

But before that, another question I encontuered was that the number of users and feature vectors are both huge. It's costly and irrelavent to put all users and feature vectors in 1 table to do matrix factorization.

I need to narrow down the subsets of users & restaurants relevant to user A.

Here I used the set of users that rate on some items that user A rated, and the set of items rated by those users. Since my feature vector contains feature ```city```, the user-item table over sets sometimes can help focus on important candidates.

## Evaluation on Models

Since we have fairly large number of users, I did rounds of model evaluations on sample users to get average precision @ 10 for each round.

For models, I tried

* baseline_model:

	Pecommend local top 10 most-rated restaurants

* als_model:

	pure als model trained on local restaurants and all previous user ratings on local restaurants

* naive_hybrid_baseline_als_model:

	Data passed to als_model is preprocessed by rating count information
	a mixture of baseline and als, aiming to make als_model more accurate on useful information

* time_biased_hybrid_model:

	Data passed to naive_hybrid_baseline_als_model is preprocessed by filtering out latest ratings in ```7 days```
	
	```7 days``` is tuned best hyperparameter I tried for a small group of users among ```1 day/ 3 days/ 7 days/ 1 month```

The best model is ```time_biased_hybrid_model```

I've improved average of ```15%``` average precision @ 10 from ```baseline_model``` to ```time_biased_hybrid_model```

The results are shown below

### Test 1
```python
rounds = 10
sample_size = 10
min_rating = 200
```
Mean Average Precision @ 10 for each round:

| Model              |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|:------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline Model     | 58% | 69% | 54% | 62% | 66% | 57% | 58% | 69% | 62% | 57% |
| Time Biased Hybrid | 68% | 83% | 63% | 70% | 62% | 83% | 80% | 75% | 80% | 67% |

Mean Average Precision @ 10 Over Samples:

| Model              | mAP@10 |
|:------------------:|:------:|
| Baseline Model     | 61%    |
| Time Biased Hybrid | 73%    |

### Test 2
```python
rounds = 10
sample_size = 10
min_rating = 100
```
Mean Average Precision @ 10 for each round:

| Model              |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|:------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline Model     | 50% | 45% | 59% | 51% | 56% | 50% | 51% | 62% | 37% | 52% |
| Time Biased Hybrid | 78% | 75% | 60% | 80% | 47% | 60% | 64% | 70% | 71% | 77% |

Mean Average Precision @ 10 Over Samples:

| Model              | mAP@10 |
|:------------------:|:------:|
| Baseline Model     | 51%    |
| Time Biased Hybrid | 68%    |

### Test 3
```python
rounds = 10
sample_size = 10
min_rating = 50
```
Mean Average Precision @ 10 for each round:

| Model              |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|:------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline Model     | 45% | 53% | 40% | 35% | 42% | 44% | 25% | 37% | 40% | 39% |
| Time Biased Hybrid | 46% | 52% | 51% | 43% | 67% | 61% | 40% | 59% | 64% | 51% |

Mean Average Precision @ 10 Over Samples:

| Model              | mAP@10 |
|:------------------:|:------:|
| Baseline Model     | 40%    |
| Time Biased Hybrid | 53%    |

### Test 4
```python
rounds = 10
sample_size = 10
min_rating = 20
```
Mean Average Precision @ 10 for each round:

| Model              |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|:------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline Model     | 32% | 35% | 20% | 29% | 29% | 38% | 51% | 23% | 22% | 25% |
| Time Biased Hybrid | 42% | 41% | 43% | 45% | 27% | 27% | 45% | 37% | 42% | 49% |

Mean Average Precision @ 10 Over Samples:

| Model              | mAP@10 |
|:------------------:|:------:|
| Baseline Model     | 30%    |
| Time Biased Hybrid | 40%    |


## Summary

* Document is important & save time in long run

* if I can do this project again I will improve this project in aspect of latency, since currently my recommend function is ad-hoc function, which cannot handle large amount of requests.

```
Number of solutions I considered are 
1. doing daily recommendation on active users and save the recommendation for next day on local

2. distribute users query to different machines

3. maybe, but i think it would be hard, to figure out a way to update model incrementally
```

* One thing I noticed from my result table is that it's not guaranteed that time biased hybrid model is always better than baseline model. 

```
After I examine my log files to look deeply into those rounds, some users get good prediction on baseline but very bad prediction on hybrid. The reason could be the trade-off of choosing ```7 days``` as hyperparameter of my hybrid.

Unfortunately, I cannot generate user distribution graph on my local machine, because the process can simply run out of memory due to the large amount of users.

If I have another chance, I can figure out a way to run this process on cloud.
```
