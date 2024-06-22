<h1 align='center'>Machine Learning Notes</h1>

<p>
<u>Title</u>: Machine Learning Notes<br>
<u>Author</u>: <a href='https://github.com/datamathur'>Utkarsh Mathur</a> <br>
<u>Summary</u>:
This is the personal ML knowledge base of <a href='https://github.com/datamathur'>Utkarsh Mathur</a> designed to store and share information in an efficient way.
</p>

<h2>ML Fundamentals</h2>



> Machine Learning as a set of methods that can automatically detect patterns in data, and then use the uncovered patterns to predict future data, or to perform other kinds of decision making under uncertainty.



- We also define Machine Learning is defined is a computer program learns from experience E with respect to some task T, if its performance P while performing task T improves over E.
- The probabilistic approach to machine learning is closely related to the field of statistics, but differ slightly in terms of its emphasis and terminology.
- Much of machine learning is concerned with devising different models, and different algorithms to fit them. However, there is no universally best model which is sometimes called the no free lunch theorem. The reason for this is that a set of assumptions that works well in one domain may work poorly in another.

<h2>ML Taxonomy</h2>

<h3>ML Problems</h3>

1. <b><u>Supervised Learning</u></b>
The goal is to learn a mapping from inputs x to outputs y, given a labeled set of input-output pairs $D\ =\ {(x_i, y_i)}$ (D is called training data and N is the number of training examples) so that the resultant model can make precise prediction on unseen data.

1. <b><u>Unsupervised Learning</u></b>
Here we are only given inputs $D\ =\ {x_i}$ and the goal is to find “interesting patterns” in the data.

1. <b><u>Semi-supervised Learning</u></b>
The goal is to extract maximum benefit from scarce labeled data while utilizing abundant unlabeled data to enhance model accuracy.

1. <b><u>Reinforcement Learning</u></b>
This is useful for learning how to act or behave when given occasional reward or punishment signals.


The general paradigm for predictive machine learning is $y\ =\ f(x)$ where f is a mathematical function used to define the machine learning model.

<h3>ML Methods</h3>

1. <b><u>Parametric Methods</u></b>
In parametric methods, we make an assumption regarding the shape of the model $f$ and train the data to fit this model by adjusting the parameters associated with the initial assumptions. This reduces the complexity of the problem at hand and increases interpretibility.

2. <b><u>Non-Parametric Methods</u></b>
Non-parametric methods donot make explicit assumptions to the functional form of $f$. Instead, they seek an estimation of $f$ that gets as close to the data points as possible without being too wiggly. 