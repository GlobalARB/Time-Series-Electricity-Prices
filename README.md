# Kvasir Energy Price Prediction Challenge
Thanks for agreeing to have a go at our recruitment challenge. We hope you 
find it interesting. If you have any questions, please feel free to reach 
out to me at phil.tromans@kvasir.ai.

## Task
Your goal is to build a forecasting model for electricity prices.
Specifically, we are asking you to investigate "day-ahead auction" prices - you
can read `Background.md` to familiarize yourself with some of the terminology.

## Getting Started
We have provided an example solution in `problem.py`.
 
Our example uses Python 3.12 and a handful of standard Python scientific libraries, which you can 
install with
```
pip install -r requirements.txt
```

You should then be able to run `python problem.py` - it will load some data, fit a simple model, and a 
couple of plots should appear.

## Solution considerations
You are welcome to use any open-source libraries or environments for your research, as long as you make
it clear how we can run your code to reproduce your results, and apply your model to data you haven't seen, e.g. to prices for the year 2025.

Our example uses RMSE to evaluate how good the forecast is on "unseen" data. You should explain how you evaluate and compare any model(s) you may investigate, and quantify how you expect your (best) model to perform on new data. 

The example provided only uses the most recent forecasts for prediction. You 
are welcome to include other features as inputs to your model.

We'd also be interested to read your thoughts on what you tried to model, 
and any interesting observations that you had while you were experimenting \- 
please don't feel obliged to write lots though. Clear, concise, code will 
  also get extra credit.

You may find something like [Jupyter](https://jupyter.org/) or 
[Google Colab](https://colab.research.google.com/) useful when analysing the 
data.

## Use of AI / LLMs
If you want to use an LLM (e.g. Gemini, ChatGPT, etc) to help craft your 
solution, then that's fine. However, if you progress to interview after 
submitting your solution, you should expect to (a) need to explain what your 
code does, and the trade-offs between different approaches and (b) be able 
to change the behaviour of your code without using an LLM in an interview 
situation. 
