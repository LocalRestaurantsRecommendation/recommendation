# From User's Perspective

Each time I clicked the recommendMe button, my personalized model will give me top 10 recommended restaurants based on my previous ratings.

My model is personalized for my location and my previous rating behaviors.

# How and why I come to the current hybrid model?

Baseline model is good, but it's not personalized enough on the input user' behaviors.

I want to improve from that using collaborative filtering.

From hybrid to time biased byhrid, there were 2 motivations:

1. I want to reduce the matrix size of ALS model.

The input of rating behaviors to ALS model is too large and overwhelm to run ALS model on my local machine.

2. I want to improve average precision @ 10 from hybrid by focusing on smaller set of previous user ratings.

If I pick the subset of rating behaviors, latest should be valued the most. 

Having this heuristic in mind, I come to the time biased hybrid solution.

# Why I choose Average Precision @ 10 to evaluate each personalized model?

Since I will recommend top 10 restaurants for each user, how good my recommendation models are is related to how effecient my recommendation models front load relevant candidates.

Thus, average precision at 10 is more appropriate to evaluate.

# What should I do if I want to remember the user's model?

model instance